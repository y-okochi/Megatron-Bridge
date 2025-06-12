from typing import List, Optional, Tuple
import re

from megatron.hub.bridge.weight_bridge import MegatronWeightBridge


class MegatronStateBridge:
    """
    Manages weight mappings between model formats with pattern matching support.
    
    This class handles collections of MegatronWeightBridge mappings and provides
    efficient pattern matching for parameter names using glob-like syntax.
    
    Args:
        *mappings: MegatronWeightBridge objects defining the mappings
    
    Example:
        weight_map = MegatronStateBridge(
            TPAwareWeightBridge(
                megatron="embedding.word_embeddings.weight",
                to="model.embed_tokens.weight",
            ),
            QKVWeightBridge(
                megatron="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
        )
        
        # Or with a list using unpacking
        mappings_list = [bridge1, bridge2, bridge3]
        weight_map = MegatronStateBridge(*mappings_list)
        
        # Get mapping for a specific parameter
        mapping = weight_map.query_megatron("decoder.layers.0.self_attention.linear_qkv.weight")
    """
    
    def __init__(self, *mappings: MegatronWeightBridge):
        """
        Initialize MegatronStateBridge with weight mappings.
        
        Args:
            *mappings: MegatronWeightBridge objects
        """
        self.mappings = list(mappings)
        
        # Pre-compile patterns for efficiency
        self._compiled_patterns = []
        self._reverse_patterns = []  # For to -> megatron lookups
        
        for mapping in mappings:
            # Compile source patterns
            if "*" in mapping.megatron:
                # Convert glob pattern to regex
                # decoder.layers.*.mlp.linear_fc1.weight -> decoder\.layers\.(\d+)\.mlp\.linear_fc1\.weight
                pattern = re.escape(mapping.megatron)
                pattern = pattern.replace(r"\*", r"(\d+)")
                self._compiled_patterns.append((re.compile(f"^{pattern}$"), mapping))
            else:
                self._compiled_patterns.append((None, mapping))
            
            # Compile destination patterns for reverse lookups
            if isinstance(mapping.to, str):
                if "*" in mapping.to:
                    pattern = re.escape(mapping.to)
                    pattern = pattern.replace(r"\*", r"(\d+)")
                    self._reverse_patterns.append((re.compile(f"^{pattern}$"), mapping))
                else:
                    self._reverse_patterns.append((None, mapping))
            else:
                # For dict destinations, compile patterns for each value
                reverse_dict_patterns = {}
                for key, to_pattern in mapping.to.items():
                    if "*" in to_pattern:
                        pattern = re.escape(to_pattern)
                        pattern = pattern.replace(r"\*", r"(\d+)")
                        reverse_dict_patterns[key] = re.compile(f"^{pattern}$")
                    else:
                        reverse_dict_patterns[key] = None
                self._reverse_patterns.append((reverse_dict_patterns, mapping))
    
    def query_megatron(self, megatron_name: str) -> Optional[MegatronWeightBridge]:
        """
        Get mapping for a megatron parameter name.
        
        This method checks both direct matches and pattern matches using
        the pre-compiled regex patterns.
        
        Args:
            megatron_name: Megatron parameter name to look up
            
        Returns:
            MegatronWeightBridge with resolved wildcards, or None if no match found
        """
        for pattern, mapping in self._compiled_patterns:
            if pattern is None:
                # Direct match
                if mapping.megatron == megatron_name:
                    return mapping
            else:
                # Pattern match
                match = pattern.match(megatron_name)
                if match:
                    # Return resolved mapping with wildcards replaced
                    return self._resolve_mapping(mapping, match.groups())
        return None
    
    def query_to(self, to_name: str) -> Optional[MegatronWeightBridge]:
        """
        Get mapping for a destination parameter name (reverse lookup).
        
        This is useful when you have a destination name and want to find
        the corresponding megatron name.
        
        Args:
            to_name: Destination parameter name to look up
            
        Returns:
            MegatronWeightBridge with resolved wildcards, or None if no match found
        """
        for pattern_info, mapping in self._reverse_patterns:
            if isinstance(mapping.to, str):
                # Simple string destination
                pattern = pattern_info
                if pattern is None:
                    # Direct match
                    if mapping.to == to_name:
                        return mapping
                else:
                    # Pattern match
                    match = pattern.match(to_name)
                    if match:
                        return self._resolve_mapping(mapping, match.groups())
            else:
                # Dict destination - check each pattern
                patterns_dict = pattern_info
                for key, pattern in patterns_dict.items():
                    if pattern is None:
                        # Direct match
                        if mapping.to[key] == to_name:
                            # Create a simplified mapping for this specific key
                            return self._resolve_mapping(mapping, ())
                    else:
                        # Pattern match
                        match = pattern.match(to_name)
                        if match:
                            return self._resolve_mapping(mapping, match.groups())
        return None
    
    def _resolve_mapping(self, mapping: MegatronWeightBridge, captures: Tuple[str, ...]) -> MegatronWeightBridge:
        """
        Resolve wildcards in mapping using captured values.
        
        Args:
            mapping: Original mapping with wildcards
            captures: Captured values from regex match
            
        Returns:
            New MegatronWeightBridge with wildcards replaced by actual values
        """
        return mapping.resolve(captures)
    
    def get_all_mappings(self) -> List[MegatronWeightBridge]:
        """Get all mappings in this MegatronStateBridge."""
        return self.mappings.copy()
    
    def get_mappings_by_pattern(self, pattern: str) -> List[MegatronWeightBridge]:
        """
        Get all mappings that match a given pattern.
        
        Args:
            pattern: Pattern to match (supports * wildcards)
            
        Returns:
            List of matching MegatronWeightBridge objects
        """
        # Convert pattern to regex
        regex_pattern = re.escape(pattern)
        regex_pattern = regex_pattern.replace(r"\*", r".*")
        compiled_pattern = re.compile(f"^{regex_pattern}$")
        
        matches = []
        for mapping in self.mappings:
            if compiled_pattern.match(mapping.megatron):
                matches.append(mapping)
        
        return matches
    
    def __len__(self) -> int:
        """Return number of mappings."""
        return len(self.mappings)
    
    def __iter__(self):
        """Iterate over mappings."""
        return iter(self.mappings)
    
    def __repr__(self) -> str:
        """String representation of MegatronStateBridge."""
        return f"MegatronStateBridge({len(self.mappings)} mappings)"
    
    def describe(self) -> str:
        """
        Get a human-readable description of all mappings.
        
        Returns:
            Formatted string describing all weight mappings
        """
        lines = [f"MegatronStateBridge with {len(self.mappings)} mappings:"]
        for i, mapping in enumerate(self.mappings):
            lines.append(f"\n{i+1}. {mapping.megatron}")
            if isinstance(mapping.to, str):
                lines.append(f"   → {mapping.to}")
            else:
                lines.append("   → {")
                for key, value in mapping.to.items():
                    lines.append(f"       {key}: {value}")
                lines.append("     }")
            
            # Show bridge type
            lines.append(f"   bridge: {type(mapping).__name__}")
        
        return "\n".join(lines)