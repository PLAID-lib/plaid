"""
Session Manager for PLAID MCP Server

Manages loaded datasets, converters, problem definitions, and conversion sessions.
"""

import uuid
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class LoadedDataset:
    """Represents a loaded PLAID dataset with its converters."""
    dataset_id: str
    local_dir: str
    datasetdict: Dict[str, Any]
    converterdict: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversionSession:
    """Represents an interactive conversion script building session."""
    session_id: str
    dataset_name: str
    components: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    
    
class SessionManager:
    """Manages state for the MCP server."""
    
    def __init__(self):
        self.datasets: Dict[str, LoadedDataset] = {}
        self.problem_definitions: Dict[str, Any] = {}
        self.conversion_sessions: Dict[str, ConversionSession] = {}
        
    # Dataset Management
    
    def add_dataset(
        self,
        local_dir: str,
        datasetdict: Dict[str, Any],
        converterdict: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a loaded dataset to the session."""
        dataset_id = str(uuid.uuid4())[:8]
        self.datasets[dataset_id] = LoadedDataset(
            dataset_id=dataset_id,
            local_dir=local_dir,
            datasetdict=datasetdict,
            converterdict=converterdict,
            metadata=metadata or {}
        )
        return dataset_id
    
    def get_dataset(self, dataset_id: str) -> Optional[LoadedDataset]:
        """Get a loaded dataset by ID."""
        return self.datasets.get(dataset_id)
    
    def list_datasets(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded datasets with basic info."""
        return {
            ds_id: {
                "local_dir": ds.local_dir,
                "splits": list(ds.datasetdict.keys()),
                "num_splits": len(ds.datasetdict)
            }
            for ds_id, ds in self.datasets.items()
        }
    
    def remove_dataset(self, dataset_id: str) -> bool:
        """Remove a dataset from the session."""
        if dataset_id in self.datasets:
            del self.datasets[dataset_id]
            return True
        return False
    
    # Problem Definition Management
    
    def add_problem_definition(self, problem_def: Any, name: Optional[str] = None) -> str:
        """Add a problem definition to the session."""
        pd_id = name or str(uuid.uuid4())[:8]
        self.problem_definitions[pd_id] = problem_def
        return pd_id
    
    def get_problem_definition(self, pd_id: str) -> Optional[Any]:
        """Get a problem definition by ID."""
        return self.problem_definitions.get(pd_id)
    
    def list_problem_definitions(self) -> Dict[str, Any]:
        """List all loaded problem definitions."""
        return {
            pd_id: {
                "name": getattr(pd, "get_name", lambda: "unnamed")(),
                "task": getattr(pd, "get_task", lambda: "unknown")()
            }
            for pd_id, pd in self.problem_definitions.items()
        }
    
    # Conversion Session Management
    
    def create_conversion_session(self, dataset_name: str) -> str:
        """Create a new conversion session."""
        from datetime import datetime
        session_id = str(uuid.uuid4())[:8]
        self.conversion_sessions[session_id] = ConversionSession(
            session_id=session_id,
            dataset_name=dataset_name,
            created_at=datetime.now().isoformat()
        )
        return session_id
    
    def get_conversion_session(self, session_id: str) -> Optional[ConversionSession]:
        """Get a conversion session by ID."""
        return self.conversion_sessions.get(session_id)
    
    def add_session_component(
        self,
        session_id: str,
        component_type: str,
        config: Dict[str, Any]
    ) -> bool:
        """Add a component to a conversion session."""
        session = self.conversion_sessions.get(session_id)
        if session:
            session.components[component_type] = config
            return True
        return False
    
    def remove_conversion_session(self, session_id: str) -> bool:
        """Remove a conversion session."""
        if session_id in self.conversion_sessions:
            del self.conversion_sessions[session_id]
            return True
        return False
