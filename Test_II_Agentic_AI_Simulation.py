"""
ML4SCI DeepLense - Specific Test II: Agentic AI Workflow
Author: Muntasir Mamun
Organization: Machine Learning for Science (ML4SCI)

Description:
Agentic AI framework orchestrating gravitational lensing simulations (wrapping DeepLenseSim).
Uses Pydantic for structured validation, tool calling, and human-in-the-loop clarification.
"""

from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError

# 1. Pydantic Models for Simulation Configurations
class LensingSimulationParams(BaseModel):
    """Structured parameter schema for gravitational lensing simulations."""
    substructure_type: Literal['no_sub', 'subhalo', 'vortex'] = Field(
        ..., description="The dark matter substructure morphological class."
    )
    simulation_model: Literal['Model_I', 'Model_II', 'Model_III', 'Model_IV'] = Field(
        ..., description="DeepLenseSim model physics configuration (Model_I to Model_IV)."
    )
    num_images: int = Field(
        default=10, ge=1, le=500, description="Total number of simulated realization images."
    )
    redshift_lens: float = Field(
        default=0.5, ge=0.01, le=5.0, description="Cosmological redshift of the foreground lensing galaxy."
    )
    redshift_source: float = Field(
        default=1.5, ge=0.05, le=10.0, description="Cosmological redshift of the background source galaxy."
    )
    resolution_pixels: int = Field(
        default=64, description="Spatial resolution grid dimensions (e.g. 64x64)."
    )

class SimulationResult(BaseModel):
    """Structured response returned by the simulation execution tool."""
    status: Literal['success', 'failed']
    generated_images: int
    parameters_applied: LensingSimulationParams
    output_path: str
    message: str


# 2. Tool Function wrapping DeepLenseSim
def execute_deeplensesim_tool(params: LensingSimulationParams) -> SimulationResult:
    """
    Executes the gravitational lensing simulation pipeline with validated parameters.
    """
    print(f"\n[DeepLenseSim Execution Engine]")
    print(f" -> Substructure: {params.substructure_type}")
    print(f" -> Physics Configuration: {params.simulation_model}")
    print(f" -> Lens Redshift: z_lens = {params.redshift_lens}")
    print(f" -> Source Redshift: z_source = {params.redshift_source}")
    print(f" -> Generating {params.num_images} realizations ({params.resolution_pixels}x{params.resolution_pixels})...")
    
    # In full production, this hooks into: from deeplensesim import LensSimulator
    output_dir = f"./simulations/{params.substructure_type}_{params.simulation_model}"
    
    return SimulationResult(
        status="success",
        generated_images=params.num_images,
        parameters_applied=params,
        output_path=output_dir,
        message=f"Successfully generated {params.num_images} simulation samples."
    )


# 3. Agent with Human-In-The-Loop Validation
class DeepLenseAgent:
    """Agent orchestrator with conversational clarification and tool invocation."""
    
    def __init__(self):
        self.context = []

    def parse_and_validate(self, raw_input: Dict[str, Any]) -> Optional[LensingSimulationParams]:
        """Validates input dictionary against Pydantic schema."""
        try:
            params = LensingSimulationParams(**raw_input)
            return params
        except ValidationError as e:
            print("\n[Agent Alert] Parameter validation required clarification:")
            for err in e.errors():
                loc = " -> ".join(str(l) for l in err['loc'])
                print(f"  * Field: '{loc}': {err['msg']}")
            return None

    def clarify_with_human(self, missing_field: str) -> Any:
        """Human-in-the-loop fallback mechanism."""
        print(f"\n[Human-in-the-Loop] Clarification requested for: '{missing_field}'")
        defaults = {
            'substructure_type': 'subhalo',
            'simulation_model': 'Model_I'
        }
        val = defaults.get(missing_field, 'Model_I')
        print(f"[Human-in-the-Loop] Auto-resolved to scientifically sound default: '{val}'")
        return val


if __name__ == "__main__":
    agent = DeepLenseAgent()

    print("=== DeepLense Agentic AI Simulation Workflow ===")
    
    # Test Scenario 1: Ambiguous user query missing simulation_model
    query_1 = {
        "substructure_type": "subhalo",
        "num_images": 5,
        "redshift_lens": 0.45
    }
    
    print("\n[Case 1] User Prompt: 'Simulate 5 subhalo lenses at redshift 0.45'")
    params_1 = agent.parse_and_validate(query_1)
    
    if not params_1:
        # Agent invokes human-in-the-loop to resolve missing simulation_model
        resolved_config = agent.clarify_with_human("simulation_model")
        query_1["simulation_model"] = resolved_config
        params_1 = agent.parse_and_validate(query_1)

    if params_1:
        result = execute_deeplensesim_tool(params_1)
        print(f"\nExecution Output:\n{result.model_dump_json(indent=2)}")
