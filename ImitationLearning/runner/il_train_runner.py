import datetime
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


class ILTrainRunner:
    """Imitation Learning trainer using supervised learning from expert trajectories."""
    def __init__(self, agent, dataset, device=None, lr=1e-4, batch_size=8, val_split=0.15, topk=(1, 2, 3)):
        """
        Initialize IL training runner.

        Args:
            agent: Agent to train
            dataset: Dataset of expert trajectories
            device: torch device (cuda or cpu)
            lr: Learning rate for Adam optimizer
            batch_size: Batch size for training
            val_split: Fraction of data to use for validation
            topk: Tuple of k values for top-k accuracy metrics
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = agent.to(self.device)

        # Get number of actions from agent's action space config
        if hasattr(agent, "use_legacy_actions") and agent.use_legacy_actions:
            self.num_actions = agent.action_space_config["num_actions"]
        else:
            # Multi-head: num_directions * num_lengths
            self.num_actions = agent.action_space_config["num_directions"] * agent.action_space_config["num_lengths"]

        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=self.seq_collate)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=self.seq_collate)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)

        # --- class weights (ignoring padding index -1) -------------------
        action_counts = np.zeros(self.num_actions)
        for _, _, tgt_act, lengths in self.train_loader:
            valid = tgt_act != -100
            for a in tgt_act[valid]:
                action_counts[a.item()] += 1
        class_w = 1.0 / (action_counts + 1e-5)
        class_w = class_w / class_w.sum() * self.num_actions
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32).to(self.device))

        self.topk = topk

    @staticmethod
    def _ensure_logits_tensor(logits):
        """
        IL expects a single-head logits tensor.
        If a dict is returned, prefer the legacy 'action' head, otherwise fall back to 'direction'.
        """
        if isinstance(logits, dict):
            if "action" in logits:
                return logits["action"]
            if "direction" in logits:
                return logits["direction"]
            raise ValueError("Received logits dict without 'action' or 'direction' key for IL training.")
        return logits

    def run(self, num_epochs=50, save_folder=None):
        """
        Run imitation learning training loop.

        Args:
            num_epochs: Number of training epochs
            save_folder: Folder to save checkpoints (if None, no saving)

        Returns:
            None
        """
        run_start = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        last_checkpoint = None
        best_val_score = -float("inf")
        best_checkpoint = None

        for epoch in range(1, num_epochs + 1):
            self.agent.train()
            tot_loss = 0.0
            train_iter = tqdm(self.train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
            for x_batch, last_act, tgt_act, lengths in train_iter:

                # Move tensors to GPU
                for k, v in x_batch.items():
                    if isinstance(v, torch.Tensor):
                        x_batch[k] = v.to(self.device)
                last_act = last_act.to(self.device)
                tgt_act = tgt_act.to(self.device)

                # Prepare batch for forward_update (expects "last_actions" key)
                x_batch["last_actions"] = last_act

                # Forward (use forward_update for sequence processing)
                # forward_update returns only logits for REINFORCE (no value head)
                logits = self.agent.forward_update(x_batch)
                logits = self._ensure_logits_tensor(logits)

                # Pack -> remove pads for CE-Loss
                pack_logits = nn.utils.rnn.pack_padded_sequence(logits, lengths.cpu(), batch_first=True, enforce_sorted=False)
                pack_tgt = nn.utils.rnn.pack_padded_sequence(tgt_act, lengths.cpu(), batch_first=True, enforce_sorted=False)

                loss = self.criterion(pack_logits.data, pack_tgt.data)
                train_iter.set_postfix(loss=loss.item())

                # Back-prop
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 5.0)
                self.optimizer.step()
                tot_loss += loss.item()

            print(f"[Epoch {epoch}] TrainLoss={tot_loss / len(self.train_loader):.4f}")
            avg_loss, acc_dict = self.evaluate()
            curr_val_score = acc_dict[1]

            # Checkpointing
            if save_folder and epoch % 25 == 0:
                checkpoint_name = f"{run_start}_imitation_agent_epoch{epoch}.pth"
                checkpoint_path = os.path.join(save_folder, checkpoint_name)
                os.makedirs(save_folder, exist_ok=True)

                if last_checkpoint and os.path.exists(last_checkpoint):
                    os.remove(last_checkpoint)

                self.save_model(checkpoint_path)
                last_checkpoint = checkpoint_path

            if save_folder and curr_val_score > best_val_score:
                best_val_score = curr_val_score

                if best_checkpoint and os.path.exists(best_checkpoint):
                    if os.path.isdir(best_checkpoint):
                        shutil.rmtree(best_checkpoint)
                    else:
                        os.remove(best_checkpoint)

                best_name = f"best_{epoch}_acc_{curr_val_score:.2f}.pth".replace(".", "_")
                best_checkpoint = os.path.join(save_folder, best_name)
                torch.save(self.agent.state_dict(), best_checkpoint)
                print(f"[INFO] New best model saved: {best_checkpoint} (acc: {curr_val_score:.2f}%)")

            torch.cuda.empty_cache()

        # Rename best model to il_pretrained_model.pth
        if save_folder and best_checkpoint and os.path.exists(best_checkpoint):
            if last_checkpoint and os.path.exists(last_checkpoint):
                os.remove(last_checkpoint)
            final_path = os.path.join(save_folder, "il_pretrained_model.pth")
            os.rename(best_checkpoint, final_path)
            print(f"[INFO] Best model renamed to: {final_path}")

    def evaluate(self):
        self.agent.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        # tqdm for evaluation
        val_iter = tqdm(self.val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for x_batch, last_act, tgt_act, lengths in val_iter:
                for key in x_batch:
                    if isinstance(x_batch[key], torch.Tensor):
                        x_batch[key] = x_batch[key].to(self.device)

                last_act = last_act.to(self.device)
                target_actions = tgt_act.to(self.device)

                # Prepare batch for forward_update
                x_batch["last_actions"] = last_act

                logits = self.agent.forward_update(x_batch)
                logits = self._ensure_logits_tensor(logits)

                # Pack -> remove pads for CE-Loss
                pack_logits = nn.utils.rnn.pack_padded_sequence(logits, lengths.cpu(), batch_first=True, enforce_sorted=False)
                pack_tgt = nn.utils.rnn.pack_padded_sequence(target_actions, lengths.cpu(), batch_first=True, enforce_sorted=False)

                loss = self.criterion(pack_logits.data, pack_tgt.data)
                total_loss += loss.item()
                val_iter.set_postfix(loss=loss.item())

                all_preds.append(pack_logits.data.cpu().detach())
                all_targets.append(pack_tgt.data.cpu().detach())

        avg_loss = total_loss / len(self.val_loader)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        acc_dict = self.compute_metrics(preds, targets)

        print(f"  Validation Loss: {avg_loss:.4f}, Top-1 Acc: {acc_dict[1]:.2f}%", end="")
        for k in self.topk:
            if k != 1:
                print(f", Top-{k} Acc: {acc_dict[k]:.2f}%", end="")
        print()
        return avg_loss, acc_dict

    def compute_metrics(self, logits, targets):
        """Compute top-k accuracies"""
        accs = {}
        max_k = max(self.topk)
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))

        for k in self.topk:
            acc = correct[:, :k].any(dim=1).float().mean().item() * 100
            accs[k] = acc
        return accs

    @staticmethod
    def seq_collate(batch):
        """
        Collate function for variable-length episode windows.
        Returns:
            x_batch  – dict with lists/tensors, shape [B,T,...]
            last_act – LongTensor [B,T]
            tgt_act  – LongTensor [B,T]
            lengths  – List[int], valid lengths before padding
        """
        obs_lists, last_list, tgt_list, lengths = zip(*batch)
        B = len(batch)
        max_T = max(lengths)

        # Occupancy Maps: just collect for now, no preprocessing
        pad_occ = []
        for obs_seq in obs_lists:
            # Occupancy not stored in current observations; fill with None
            occ_seq = [None for _ in obs_seq]
            pad_occ.append(occ_seq + [None] * (max_T - len(occ_seq)))
        # shape: [B, T], each entry raw occ_map

        # RGB frames: no ToTensor, just collect
        pad_rgb = []
        for obs_seq in obs_lists:
            rgb_seq = [o.state[0] for o in obs_seq]
            pad_rgb.append(rgb_seq + [None] * (max_T - len(rgb_seq)))
        # shape: [B, T], each entry np.ndarray or PIL image

        # Depth frames: collect from state[1]
        pad_depth = []
        for obs_seq in obs_lists:
            depth_seq = [o.state[1] if len(o.state) > 1 else None for o in obs_seq]
            pad_depth.append(depth_seq + [None] * (max_T - len(depth_seq)))
        # shape: [B, T], each entry np.ndarray

        # LSSG and GSSG: just pass lists
        pad_lssg = []
        pad_gssg = []
        pad_lssg_mask = []
        pad_gssg_mask = []
        for obs_seq in obs_lists:
            # State layout: [rgb, depth, local_sg, global_sg]
            lssg_seq = [o.state[2] if len(o.state) > 2 else None for o in obs_seq]
            gssg_seq = [o.state[3] if len(o.state) > 3 else None for o in obs_seq]
            pad_len = max_T - len(lssg_seq)
            pad_lssg.append(lssg_seq + [None] * pad_len)
            pad_lssg_mask.append([1] * len(lssg_seq) + [0] * pad_len)
            pad_gssg.append(gssg_seq + [None] * pad_len)
            pad_gssg_mask.append([1] * len(gssg_seq) + [0] * pad_len)

        # Actions
        # For legacy actions: shape [B, T] -> convert to [B, T, 2] (second column is dummy)
        last_act_1d = torch.tensor([list(la) + [-100] * (max_T - len(la)) for la in last_list], dtype=torch.long)
        # Add dummy second column for compatibility with multi-head action format
        last_act = torch.stack([last_act_1d, torch.zeros_like(last_act_1d)], dim=-1)  # [B, T, 2]

        tgt_act = torch.stack(
            [torch.cat([torch.tensor(ta, dtype=torch.long), torch.full((max_T - len(ta),), -100, dtype=torch.long)]) for ta in tgt_list]
        )

        # Agent positions (if available)
        pad_agent_pos = []
        for obs_seq in obs_lists:
            agent_seq = [o.info.get("agent_pos", None) for o in obs_seq]
            pad = [None] * (max_T - len(agent_seq))
            pad_agent_pos.append(agent_seq + pad)
        # Can be None, preprocessing handles it!

        x_batch = {
            "occupancy": pad_occ,  # [B, T]
            "rgb": pad_rgb,  # [B, T]
            "depth": pad_depth,  # [B, T]
            "lssg": pad_lssg,  # [B, T]
            "lssg_mask": pad_lssg_mask,
            "gssg": pad_gssg,  # [B, T]
            "gssg_mask": pad_gssg_mask,
            "agent_pos": pad_agent_pos,  # [B, T]
        }

        return x_batch, last_act, tgt_act, torch.tensor(lengths)

    def _get_encoder_weights(self):
        return [p.detach().clone().cpu() for p in self.agent.encoder.parameters() if p.requires_grad]

    def _get_encoder_weight_change(self, prev_weights):
        current_weights = self._get_encoder_weights()
        diffs = [((c - p) ** 2).sum().sqrt().item() for c, p in zip(current_weights, prev_weights)]
        return np.mean(diffs)

    def _get_encoder_grad_norm(self):
        total_norm = 0.0
        for p in self.agent.encoder.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm**0.5

    def save_model(self, path):
        torch.save(self.agent.state_dict(), path)
        print(f"Model saved to {path}")
