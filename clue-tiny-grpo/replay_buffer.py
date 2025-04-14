import torch
import sys

# Import Self conditionally for Python < 3.11
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
    
from dataclasses import dataclass, fields
from typing import Optional, Literal

import torch.nn.functional as F


def zero_pad_sequences(
    sequences: list[torch.Tensor], side: Literal["left", "right"] = "right"
) -> torch.Tensor:
    """
    Pad a list of variable length sequences to the same length.
    
    Args:
        sequences: List of tensors with different last dimensions
        side: Which side to pad on (left or right)
    
    Returns:
        A single tensor with padded sequences
    """
    if not sequences:
        print("Warning: Empty sequence list provided to zero_pad_sequences")
        return torch.empty(0)
    
    # Validate input - make sure all are tensors
    if not all(isinstance(seq, torch.Tensor) for seq in sequences):
        non_tensors = [i for i, seq in enumerate(sequences) if not isinstance(seq, torch.Tensor)]
        print(f"Error: Non-tensor items at indices {non_tensors}")
        return torch.empty(0)
    
    # 1. Group by dimensionality to handle similar tensors together
    dim_groups = {}
    for i, seq in enumerate(sequences):
        ndim = seq.ndim
        if ndim not in dim_groups:
            dim_groups[ndim] = []
        dim_groups[ndim].append((i, seq))
    
    if len(dim_groups) > 1:
        # Process each group separately and return the largest group's result
        max_group_dim = max(dim_groups.keys(), key=lambda k: len(dim_groups[k]))
        print(f"Warning: Mixed tensor dimensions in zero_pad_sequences. Processing {len(dim_groups[max_group_dim])}/{len(sequences)} tensors with dim={max_group_dim}")
        filtered_sequences = [seq for _, seq in dim_groups[max_group_dim]]
        return zero_pad_sequences(filtered_sequences, side)
    
    # Now all sequences have the same dimensionality
    common_ndim = list(dim_groups.keys())[0]
    
    # 2. For tensors with more than 1 dimension, check first dimension consistency
    if common_ndim > 1:
        # Group by first dimension size
        first_dim_groups = {}
        for i, seq in dim_groups[common_ndim]:
            first_dim = seq.size(0)
            if first_dim not in first_dim_groups:
                first_dim_groups[first_dim] = []
            first_dim_groups[first_dim].append((i, seq))
        
        if len(first_dim_groups) > 1:
            # Process the largest group
            max_first_dim = max(first_dim_groups.keys(), key=lambda k: len(first_dim_groups[k]))
            print(f"Warning: Inconsistent first dimensions. Processing {len(first_dim_groups[max_first_dim])}/{len(sequences)} tensors with first_dim={max_first_dim}")
            filtered_sequences = [seq for _, seq in first_dim_groups[max_first_dim]]
            return zero_pad_sequences(filtered_sequences, side)
    
    # At this point we have tensors with consistent dimensionality and first dimension
    filtered_sequences = [seq for _, seq in dim_groups[common_ndim]]
    
    # If dealing with 3D tensors, handle batch dimension specially
    if common_ndim == 3:
        # For 3D tensors with shape [batch, group, seq_len]
        # First ensure consistent second dimension (group size)
        group_sizes = [seq.size(1) for seq in filtered_sequences]
        if len(set(group_sizes)) > 1:
            max_group_size = max(group_sizes)
            padded_group_sequences = []
            
            for seq in filtered_sequences:
                if seq.size(1) < max_group_size:
                    # Pad the group dimension
                    batch_size, group_size, seq_len = seq.shape
                    padding = torch.zeros(batch_size, max_group_size - group_size, seq_len, 
                                         dtype=seq.dtype, device=seq.device)
                    padded_seq = torch.cat([seq, padding], dim=1)
                    padded_group_sequences.append(padded_seq)
                else:
                    padded_group_sequences.append(seq)
            
            filtered_sequences = padded_group_sequences
        
        # Now reshape to [batch*group, seq_len] for padding along sequence dimension
        flattened_sequences = []
        for seq in filtered_sequences:
            batch_size, group_size, seq_len = seq.shape
            flattened_sequences.append(seq.reshape(batch_size * group_size, seq_len))
        
        filtered_sequences = flattened_sequences
    
    # Find max sequence length
    if all(seq.dim() > 0 for seq in filtered_sequences):
        max_len = max(seq.size(-1) for seq in filtered_sequences)
    else:
        print("Error: Cannot determine sequence length for scalar tensors")
        return torch.empty(0)
    
    padded_sequences = []
    target_dtype = filtered_sequences[0].dtype
    
    try:
        for i, seq in enumerate(filtered_sequences):
            pad_len = max_len - seq.size(-1)
            if pad_len < 0:
                print(f"Warning: Sequence {i} length {seq.size(-1)} exceeds max_len {max_len}. Truncating.")
                # Truncate instead of skipping to maintain batch size
                seq = seq[..., :max_len]
                pad_len = 0
                
            if pad_len == 0:
                padded_sequences.append(seq)
                continue
                
            padding = (pad_len, 0) if side == "left" else (0, pad_len)
            # Pad based on the last dimension
            pad_value = 0 if target_dtype != torch.bool else False
            
            # Handle different dimensions
            if seq.ndim > 1:
                num_prefix_dims = seq.ndim - 1
                full_padding = (0, 0) * num_prefix_dims + padding
                padded_seq = F.pad(seq, full_padding, value=pad_value)
            else:
                padded_seq = F.pad(seq, padding, value=pad_value)
                
            padded_sequences.append(padded_seq.to(target_dtype))
            
        # If all padded successfully, stack them
        result = torch.stack(padded_sequences, dim=0)
        
        # For 3D tensors that were flattened, reshape back to original form
        if common_ndim == 3 and result.dim() == 2:
            # Calculate original batch and group sizes
            batch_size = len(dim_groups[common_ndim])
            group_size = result.size(0) // batch_size
            # Reshape to [batch, group, seq_len]
            result = result.reshape(batch_size, group_size, result.size(1))
            
        return result
        
    except Exception as e:
        print(f"Error during padding in zero_pad_sequences: {e}")
        
        # Print detailed shape information
        print("Shapes of sequences:")
        for i, seq in enumerate(filtered_sequences):
            print(f"  Item {i}: {seq.shape}, dtype: {seq.dtype}")
            
        # Return empty tensor to indicate failure
        return torch.empty(0, dtype=target_dtype)


@dataclass
class Experience:
    """A single experience step collected from the environment."""
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    log_probs_ref: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    action_mask: torch.Tensor
    kl: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> Self:
        members = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[field.name] = v
        return Experience(**members)


def split_experience_batch(experience: Experience) -> list[Experience]:
    batch_size = experience.sequences.size(0)
    batch_data = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "log_probs_ref",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            vals = [None] * batch_size
        else:
            vals = torch.unbind(value)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_data[i][key] = v

    return [Experience(**data) for data in batch_data]


def join_experience_batch(items: list[Experience | dict]) -> dict[str, Optional[torch.Tensor]]:
    """Join list of experiences or dicts into a single dictionary of batched tensors, handling padding."""
    result: dict[str, Optional[torch.Tensor]] = {}
    if not items:
        return result

    # Validate input batch
    for i, item in enumerate(items):
        if not isinstance(item, (Experience, dict)):
            print(f"Error: Item {i} in batch is not a dictionary or Experience: {type(item)}")
            return {}  # Return empty dict if batch contains invalid items

    # Determine keys based on the expected Experience structure or dict keys
    # Use Experience fields as the canonical list of expected keys
    keys = [f.name for f in fields(Experience)] 
    # Define which keys need padding (variable sequence length)
    keys_to_pad = {"sequences", "action_log_probs", "log_probs_ref", "action_mask", "attention_mask"}
    # Define keys that might be None and should be handled
    optional_keys = {"returns", "advantages", "log_probs_ref", "kl", "attention_mask"}

    # Check the type of the first item to decide how to access values
    access_method = getattr if isinstance(items[0], Experience) else lambda item, key: item.get(key)

    # Dictionary to track tensor dimensions for consistent reshaping
    dim_info = {}
    
    for key in keys:
        try:
            # Get all values for the current key using the correct access method
            vals = [access_method(item, key) for item in items]
            
            # Handle optional keys that might be None
            if key in optional_keys and all(v is None for v in vals):
                result[key] = None
                continue
            
            # Filter out None values if not all are None
            if any(v is None for v in vals):
                non_none_vals = [v for v in vals if v is not None]
                if not non_none_vals:
                    result[key] = None
                    continue
                print(f"Warning: Mixed None/Tensor values for key '{key}'. Using {len(non_none_vals)}/{len(vals)} non-None values.")
                vals = non_none_vals
            
            # Check that all values are tensors
            if not all(isinstance(v, torch.Tensor) for v in vals):
                print(f"Error: Non-tensor values for key '{key}'")
                result[key] = None
                continue
            
            # Group tensors by dimensionality for more accurate handling
            dim_groups = {}
            for i, tensor in enumerate(vals):
                ndim = tensor.dim()
                if ndim not in dim_groups:
                    dim_groups[ndim] = []
                dim_groups[ndim].append((i, tensor))
            
            # If we have multiple dimension groups, use the most common one
            if len(dim_groups) > 1:
                most_common_dim = max(dim_groups.items(), key=lambda x: len(x[1]))[0]
                print(f"Warning: Mixed dimensions for key '{key}'. Using {len(dim_groups[most_common_dim])}/{len(vals)} tensors with dim={most_common_dim}.")
                vals = [t for _, t in dim_groups[most_common_dim]]
                
            # Collect shape information for consistent handling later
            if vals and isinstance(vals[0], torch.Tensor):
                if key not in dim_info:
                    dim_info[key] = {
                        "dim": vals[0].dim(),
                        "batch_size": vals[0].size(0) if vals[0].dim() > 0 else 1,
                        "shape": vals[0].shape
                    }
            
            # Pad sequences or stack tensors based on their structure
            if key in keys_to_pad and any(v.dim() > 1 for v in vals):
                # For padded sequences, ensure consistent dimensions before padding
                try:
                    # First identify if we're dealing with 2D, 3D, or higher sequences
                    ndim = vals[0].dim()
                    
                    # For 2D sequences (batch, seq_len)
                    if ndim == 2:
                        # This is likely (batch, seq_len) -> pad along seq_len dimension
                        result[key] = zero_pad_sequences(vals, side="left")
                    # For 3D sequences (batch, group_size, seq_len)
                    elif ndim == 3:
                        # Track the group sizes for reshaping
                        group_sizes = [v.size(1) for v in vals]
                        if len(set(group_sizes)) > 1:
                            # If group sizes differ, pad to the largest group size
                            max_group = max(group_sizes)
                            padded_vals = []
                            for v in vals:
                                if v.size(1) < max_group:
                                    # Pad along group dimension
                                    padding = torch.zeros(v.size(0), max_group - v.size(1), v.size(2), 
                                                          dtype=v.dtype, device=v.device)
                                    padded_vals.append(torch.cat([v, padding], dim=1))
                                else:
                                    padded_vals.append(v)
                            vals = padded_vals
                        
                        # Now pad along sequence dimension
                        # First reshape to (batch*group_size, seq_len) for padding
                        reshaped_vals = [v.reshape(-1, v.size(-1)) for v in vals]
                        padded = zero_pad_sequences(reshaped_vals, side="left")
                        
                        # Store with the padding result directly
                        result[key] = padded
                    else:
                        # Handle higher dimensions if needed
                        print(f"Warning: Complex tensor with {ndim} dimensions for key '{key}'. Attempting basic padding.")
                        result[key] = zero_pad_sequences(vals, side="left")
                        
                except Exception as e:
                    print(f"Error padding sequences for key '{key}': {e}")
                    # Provide detailed shape information for debugging
                    print("Shapes:")
                    for i, v in enumerate(vals):
                        print(f"  Item {i}: {v.shape}")
                    result[key] = None
            else:
                # Handle non-padded tensors (scalar values or fixed-size tensors)
                try:
                    if vals[0].dim() == 0:  # scalars
                        result[key] = torch.stack(vals)
                    else:
                        # First check if all shapes match
                        shapes = [v.shape for v in vals]
                        if len(set(shapes)) > 1:
                            print(f"Warning: Inconsistent shapes for key '{key}': {shapes}")
                            # Try to reshape to common size if possible
                            first_dims = [s[0] for s in shapes]
                            if len(set(first_dims)) == 1:
                                # Only first dimension matches, can still stack
                                result[key] = torch.stack([v.view(v.size(0), -1) for v in vals], dim=0)
                            else:
                                # Cannot safely combine, try concatenation as fallback
                                result[key] = torch.cat(vals, dim=0)
                        else:
                            # All shapes match, stack them
                            result[key] = torch.stack(vals, dim=0)
                except RuntimeError as e:
                    print(f"Error stacking tensors for key '{key}': {e}")
                    # Try concatenation as a fallback
                    try:
                        result[key] = torch.cat(vals, dim=0)
                    except RuntimeError:
                        print(f"Error concatenating tensors for key '{key}' as fallback.")
                        result[key] = None
                
        except Exception as e:
            print(f"Error joining batch for key '{key}': {e}")
            result[key] = None  # Ensure key exists but is None if error occurred
            continue
    
    # Ensure all expected keys are present in the result, even if None
    for key in keys:
        if key not in result:
            result[key] = None
    
    # Apply consistent reshaping based on collected dimension information
    if dim_info and any(result[k] is not None for k in keys):
        try:
            # Find a reference tensor with 3D shape to guide reshaping
            ref_key = next((k for k in keys_to_pad if k in dim_info and dim_info[k]["dim"] == 3), None)
            
            if ref_key and ref_key in result and result[ref_key] is not None:
                ref_tensor = result[ref_key]
                batch_size = ref_tensor.size(0)
                
                # Reshape other tensors to match if needed
                for key in result:
                    if result[key] is None or key == ref_key:
                        continue
                        
                    tensor = result[key]
                    if tensor.dim() == 2 and tensor.size(0) != batch_size:
                        # Tensor might need reshaping to match batch size
                        print(f"Reshaping tensor for key '{key}' from shape {tensor.shape} to match batch size {batch_size}")
                        if tensor.size(0) > batch_size and tensor.size(0) % batch_size == 0:
                            # This is likely a flattened tensor that needs reshaping
                            group_size = tensor.size(0) // batch_size
                            result[key] = tensor.reshape(batch_size, group_size, -1)
                        elif tensor.size(0) < batch_size and batch_size % tensor.size(0) == 0:
                            # Need to repeat this tensor to match batch size
                            repeat_factor = batch_size // tensor.size(0)
                            result[key] = tensor.repeat_interleave(repeat_factor, dim=0)
        except Exception as e:
            print(f"Error during tensor reshaping phase: {e}")
    
    return result


class ReplayBuffer:
    def __init__(self, limit: int = 0) -> None:
        self.limit = limit
        self.items: list[Experience] = []

    def append(self, experience: Experience) -> None:
        items = split_experience_batch(experience)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Experience:
        return self.items[idx]
