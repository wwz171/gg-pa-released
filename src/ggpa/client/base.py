"""Client-side base definitions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ggpa.core.interfaces import Client
from ggpa.core.protocol import ClientRequest, ClientReply

class ClientBase(ABC, Client):
    """User-friendly Client base class with minimal implementation requirements.
    
    DESIGN PHILOSOPHY:
    ==================
    Base provides all Request/Reply handling logic. Users only implement:
        1. denoise_sample() - The CORE denoising method
        2. Set projector and forward_process attributes
    
    Everything else (gradient computation, log_prob, properties, request handling)
    is automatically handled by the base class!
    
    REQUIRED ATTRIBUTES:
    ====================
        client_id (str): Unique identifier for this client
        projector (ProjectorBase): Signal projection Φ: s → y
        forward_process (ForwardProcessBase): Forward diffusion q_t_diff(y | x)
    
    REQUIRED METHODS:
    =================
        denoise_sample(y, t_diff, seed): The ONLY method users must implement!
    
    SUPPORTED REQUEST TYPES:
    ========================
        - 'sample': Returns denoised x ~ p(x | y=Φ(s), t_diff)
        - 'gradient': Returns ∇_s log q_t_diff(y | x) via chain rule
        - 'log_prob': Returns log q_t_diff(y | x)
        - 'properties': Returns client metadata
    
    MINIMAL USER CODE EXAMPLE:
    ==========================
        class MyClient(ClientBase):
            def __init__(self, client_id, model):
                self.client_id = client_id
                self.model = model
                self.projector = IdentityProjector()
                self.forward_process = GaussianForwardProcess()
            
            def denoise_sample(self, y, t_diff, seed=None):
                # That's it! Just denoise y given t_diff
                return self.model.denoise(y, t_diff, seed)
    
    ADVANCED: Overriding default behavior
    ======================================
    Users can optionally override these methods for custom behavior:
        - compute_log_prob(y, x, t_diff): Custom log prob computation
        - get_public_properties(): Control which properties are exposed to server
        - _build_all_properties(): Add custom properties beyond defaults
        - compute_gradient(y, x, t_diff, s): Custom gradient computation
        - get_properties(): Custom metadata
        - handle_request(request): Full custom request handling
    """
    
    # Required attributes (subclasses must set these)
    client_id: str
    projector: 'ProjectorBase'
    forward_process: 'ForwardProcessBase'

    # ========== Cached clean state ==========

    @property
    def current_x(self) -> Optional[Any]:
        """Cached denoised sample from the last 'sample' request.

        Set automatically when the client processes a 'sample' request.
        Used by 'log_prob' and 'gradient' requests when 'sample' is not
        co-requested, so that reduced_potential evaluations do NOT trigger
        an expensive (and semantically wrong) re-denoise.

        Can also be set externally – e.g. in Replica Exchange to inject
        per-replica x before calling kernel.reduced_potential().
        """
        return getattr(self, '_current_x', None)

    @current_x.setter
    def current_x(self, value: Any) -> None:
        self._current_x = value

    # ========== REQUIRED: User must implement this ==========
    
    @abstractmethod
    def denoise_sample(self, y: Any, t_diff: float, seed: Optional[int] = None) -> Any:
        """The CORE method: denoise observation y given noise level t_diff.
        
        This is the ONLY method users MUST implement!
        
        Args:
            y: Projected observation y = Φ(s)
            t_diff: Diffusion time t_diff ∈ [0, 1]
            seed: Optional random seed for deterministic sampling
        
        Returns:
            x: Denoised sample (same space as y)
        
        Example:
            def denoise_sample(self, y, t_diff, seed=None):
                return self.model.denoise(y, t_diff, seed)
        """
        pass

    # ========== DEFAULT IMPLEMENTATIONS: Users rarely need to override ==========
    
    def compute_log_prob(self, y: Any, x: Any, t_diff: float) -> Any:
        """Compute log q_t_diff(y | x) using forward_process.
        
        Default implementation: Calls self.forward_process.log_q_fwd()
        
        Args:
            y: Noisy observation
            x: Clean sample
            t_diff: Diffusion time
        
        Returns:
            Log probability (can be scalar, array, or any type)
        """
        return self.forward_process.log_q_fwd(y, x, t_diff)
    
    def compute_gradient(self, y: Any, x: Any, t_diff: float, s: Any) -> Optional[Any]:
        """Compute gradient ∇_s log q_t_diff(y | x) using chain rule.
        
        Default implementation:
            1. Compute ∇_y log q_t_diff(y | x) via forward_process.grad_log_q_fwd()
            2. Apply chain rule: ∇_s = (∂Φ/∂s)^T @ ∇_y via projector.backprop_gradient()
        
        Returns None if forward_process doesn't support gradients.
        
        Args:
            y: Noisy observation
            x: Clean sample
            t_diff: Diffusion time
            s: Signal (needed for chain rule)
        
        Returns:
            Gradient ∇_s log q_t_diff(y | x), or None if not supported
        """
        # Step 1: Compute ∇_y log q_t_diff(y | x)
        grad_y = self.forward_process.grad_log_q_fwd(y, x, t_diff)
        if grad_y is None:
            return None
        
        # Step 2: Chain rule: ∇_s = (∂Φ/∂s)^T @ ∇_y
        grad_s = self.projector.backprop_gradient(s, grad_y)
        return grad_s
    
    def _build_all_properties(self) -> Dict[str, Any]:
        """Build dictionary of all available client properties.
        
        This internal method gathers all properties that CAN be exposed.
        Override this to add custom properties beyond the defaults.
        
        Default properties:
            - client_id: Unique client identifier (ALWAYS included)
            - projector_type: Name of projector class
            - forward_process_type: Name of forward process class
        
        Returns:
            Dictionary of all available properties
        
        Example:
            class MyClient(ClientBase):
                def _build_all_properties(self):
                    props = super()._build_all_properties()
                    props['custom_field'] = self.my_custom_value
                    return props
        """
        return {
            "client_id": self.client_id,
            "projector_type": type(self.projector).__name__,
            "forward_process_type": type(self.forward_process).__name__,
        }
    
    def get_public_properties(self) -> Dict[str, Any]:
        """Get properties that this client is willing to expose publicly.
        
        Override this method to control which properties are exposed to the server.
        
        Default: Exposes ALL available properties (client_id + projector_type + forward_process_type).
        
        Returns:
            Dictionary of public properties
        
        Example - Expose only specific properties:
            class MyClient(ClientBase):
                def get_public_properties(self):
                    all_props = self._build_all_properties()
                    # Only expose client_id and projector_type
                    return {
                        'client_id': all_props['client_id'],
                        'projector_type': all_props['projector_type']
                    }
        
        Example - Add custom properties:
            class MyClient(ClientBase):
                def _build_all_properties(self):
                    props = super()._build_all_properties()
                    props['model_name'] = 'my_diffusion_v2'
                    props['dataset'] = 'imagenet'
                    return props
                
                def get_public_properties(self):
                    # Expose everything (default behavior)
                    return self._build_all_properties()
        """
        return self._build_all_properties()
    
    def get_properties(self, property_names: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """Return requested client properties.
        
        This method queries the public properties exposed by get_public_properties().
        Server uses this method via 'properties' request type.
        
        Args:
            property_names: Optional list of specific properties to return.
                          If None, returns all public properties.
                          If a property is not found, returns None for that property.
        
        Returns:
            Dictionary with requested client properties
            
        Note:
            - To control which properties are exposed, override get_public_properties()
            - To add custom properties, override _build_all_properties()
        """
        # Get all public properties
        all_properties = self.get_public_properties()
        
        # Validate: client_id must ALWAYS be present
        if "client_id" not in all_properties:
            from ggpa.core.logging import get_logger
            logger = get_logger("client")
            logger.warning(
                f"Client public properties missing 'client_id'. Adding it automatically."
            )
            all_properties["client_id"] = self.client_id
        
        # If no specific properties requested, return all public properties
        if property_names is None:
            return all_properties
        
        # Convert single property to list
        if isinstance(property_names, str):
            property_names = [property_names]
        
        # Extract requested properties
        from ggpa.core.logging import get_logger
        logger = get_logger("client")
        
        result = {}
        for prop_name in property_names:
            if prop_name in all_properties:
                result[prop_name] = all_properties[prop_name]
            else:
                result[prop_name] = None
                logger.warning(
                    f"Client {self.client_id}: Property '{prop_name}' not found. "
                    f"Available public properties: {list(all_properties.keys())}"
                )
        
        return result
    
    def handle_request(self, request: ClientRequest) -> ClientReply:
        """Handle client request (DEFAULT IMPLEMENTATION).
        
        Users typically DO NOT need to override this method!
        
        This method automatically:
        1. Projects signal: y = Φ(s)
        2. Dispatches to appropriate methods based on request_types
        3. Handles batch requests
        4. Sets correct status codes
        5. Catches and reports errors
        
        Args:
            request: ClientRequest from server
        
        Returns:
            ClientReply with results or error information
        """
        try:
            # 1. Project signal s → y
            y = self.projector.forward(request.s)
            
            # 2. Parse request_types (can be str or List[str])
            types = request.request_types
            if isinstance(types, str):
                types = [types]
            
            # 3. Process each request type
            data = {}
            for req_type in types:
                if req_type == 'sample':
                    # Generate sample and cache it
                    x = self.denoise_sample(y, request.t_diff, request.seed)
                    data['sample'] = x
                    self._current_x = x          # ← cache for later log_prob / gradient
                
                elif req_type == 'gradient':
                    # Use sample from this batch, or cached current_x, or denoise
                    if 'sample' not in data:
                        if self.current_x is not None:
                            x = self.current_x
                        else:
                            x = self.denoise_sample(y, request.t_diff, request.seed)
                            self._current_x = x
                        data['sample'] = x
                    else:
                        x = data['sample']
                    
                    # Compute gradient (may return None if not supported)
                    grad = self.compute_gradient(y, x, request.t_diff, request.s)
                    data['gradient'] = grad
                
                elif req_type == 'log_prob':
                    # Use sample from this batch, or cached current_x, or denoise
                    if 'sample' not in data:
                        if self.current_x is not None:
                            x = self.current_x
                        else:
                            x = self.denoise_sample(y, request.t_diff, request.seed)
                            self._current_x = x
                        data['sample'] = x
                    else:
                        x = data['sample']
                    
                    # Compute log probability
                    log_prob = self.compute_log_prob(y, x, request.t_diff)
                    data['log_prob'] = log_prob
                
                elif req_type == 'properties':
                    # Get client metadata
                    # Check if specific properties are requested in metadata
                    property_names = None
                    if hasattr(request, 'metadata') and request.metadata:
                        property_names = request.metadata.get('property_names', None)
                    data['properties'] = self.get_properties(property_names)
                
                else:
                    # Unsupported request type
                    data[req_type] = None
            
            # 4. Determine status code
            # Special handling for 'properties' - check if any property value is None
            has_partial_properties = False
            if 'properties' in data and isinstance(data['properties'], dict):
                if any(v is None for v in data['properties'].values()):
                    has_partial_properties = True
            
            # Check all data values (excluding properties dict internals)
            all_success = all(v is not None for v in data.values()) and not has_partial_properties
            any_success = any(v is not None for v in data.values())
            
            if all_success:
                status = 'success'
            elif any_success:
                status = 'partial'
            else:
                status = 'unsupported'
            
            # 5. Return successful reply
            return ClientReply(
                client_id=self.client_id,
                request_id=request.request_id,
                status_code=status,
                error=None,
                data=data
            )
        
        except Exception as e:
            # 6. Handle errors gracefully
            return ClientReply(
                client_id=self.client_id,
                request_id=request.request_id,
                status_code='error',
                error=str(e),
                data={}
            )



class ProjectorBase(ABC):
    """Abstract base class for projector implementations.
    
    Projectors map the global signal s to observation space y = Φ(s).
    
    CRITICAL REQUIREMENT: All projectors MUST implement backprop_gradient()
    for gradient-based aggregation via chain rule.
    
    Chain rule: ∇_s f(Φ(s)) = (∂Φ/∂s)^T @ ∇_y f(y)
    
    Required methods:
        forward(s): Project signal to observation space
        backprop_gradient(s, grad_y): Apply chain rule for gradient backpropagation
    
    Example (Linear projector):
        class MyLinearProjector(ProjectorBase):
            def __init__(self, matrix):
                self.P = matrix  # Projection matrix
            
            def forward(self, s):
                return self.P @ s
            
            def backprop_gradient(self, s, grad_y):
                return self.P.T @ grad_y  # Chain rule for linear projection
    """
    
    @abstractmethod
    def forward(self, s: Any) -> Any:
        """Project the signal into observation space: y = Φ(s).
        
        Args:
            s: Signal (any format for flexibility)
        
        Returns:
            y: Projected observation
        """
        pass
    
    @abstractmethod
    def backprop_gradient(self, s: Any, grad_y: Any) -> Any:
        """Chain rule gradient: ∇_s f(Φ(s)) = (∂Φ/∂s)^T @ grad_y.
        
        REQUIRED METHOD: Must be implemented for gradient-based aggregation.
        
        For linear projections y = P @ s:
            ∇_s = P.T @ grad_y
        
        For nonlinear projections y = Φ(s):
            ∇_s = J_Φ(s).T @ grad_y  (where J is Jacobian)
        
        Args:
            s: Signal
            grad_y: Gradient in observation space ∇_y f(y)
        
        Returns:
            grad_s: Gradient in signal space ∇_s f(Φ(s))
        """
        pass
    
    # Legacy method - kept for backward compatibility
    def grad_forward(self, s: Any) -> Optional[Any]:
        """Compute Jacobian ∇_s Φ(s) (DEPRECATED).
        
        DEPRECATED: Use backprop_gradient() instead.
        
        This method is kept for backward compatibility.
        New code should implement backprop_gradient() directly.
        """
        
        return None


class ForwardProcessBase(ABC):
    """Abstract base class for forward diffusion process implementations.
    
    Forward process defines how clean samples x are corrupted to observations y.
    Typically: q_t_diff(y | x) = N(y; α(t_diff)x, σ²(t_diff)I) for Gaussian noise.
    
    Required methods:
        log_q_fwd(y, x, t_diff): Log density log q_t_diff(y | x)
        alpha(t_diff): Signal coefficient α(t_diff)
        sigma(t_diff): Noise standard deviation σ(t_diff)
    
    Required methods:
        grad_log_q_fwd(y, x, t_diff): Gradient ∇_y log q_t_diff(y | x)
    
    Example (Gaussian forward process):
        class GaussianForward(ForwardProcessBase):
            def log_q_fwd(self, y, x, t_diff):
                alpha = self.alpha(t_diff)
                sigma = self.sigma(t_diff)
                # log N(y; alpha*x, sigma^2)
                return -0.5 * np.sum((y - alpha * x)**2) / sigma**2
            
            def alpha(self, t_diff):
                return np.sqrt(1 - t_diff)
            
            def sigma(self, t_diff):
                return np.sqrt(t_diff)
    """
    
    @abstractmethod
    def log_q_fwd(self, y: Any, x: Any, t_diff: float) -> Any:
        """Compute log q_t_diff(y | x).
        
        Used for:
        1. Reduced potential computation
        2. MCMC acceptance ratios
        
        Args:
            y: Noisy observation
            x: Clean sample
            t_diff: Diffusion time in [0, 1]
            
        Returns:
            Log probability (can be scalar, array, or any type)
            Unnormalized constants are allowed.
        """
        pass
    
    @abstractmethod
    def alpha(self, t_diff: float) -> Any:
        """Signal coefficient α(t_diff) in diffusion process.
        
        For DDPM-style: α(t_diff) = √(1 - t_diff)
        
        Args:
            t_diff: Diffusion time
            
        Returns:
            Signal coefficient (typically scalar, but can be any type)
        """
        pass
    
    @abstractmethod
    def sigma(self, t_diff: float) -> Any:
        """Noise standard deviation σ(t_diff) in diffusion process.
        
        For DDPM-style: σ(t_diff) = √t_diff
        
        Args:
            t_diff: Diffusion time
            
        Returns:
            Noise standard deviation (typically scalar, but can be any type)
        """
        pass
    
    @abstractmethod
    def grad_log_q_fwd(self, y: Any, x: Any, t_diff: float) -> Any:
        """Compute gradient ∇_y log q_t_diff(y | x).
        
        REQUIRED METHOD for gradient-based aggregation!
        
        Used for:
        1. Gradient-based MCMC aggregators
        2. Score matching
        3. Chain rule computation in ClientBase.compute_gradient()
        
        For Gaussian forward process:
            q_t_diff(y | x) = N(y; α(t_diff)x, σ²(t_diff)I)
            ∇_y log q_t_diff(y | x) = -(y - α(t_diff)x) / σ²(t_diff)
        
        Args:
            y: Noisy observation
            x: Clean sample
            t_diff: Diffusion time
            
        Returns:
            Gradient with same shape as y
            Should match the data type (NumPy/PyTorch/JAX/etc.)
        """
        pass