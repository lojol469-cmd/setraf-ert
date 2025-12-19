"""
OUTILS GÉOLOGIQUES POUR KIBALI
Module d'initialisation du package tools
"""

from .web_search_tools import GeologyWebSearchTool, WebResearchManager
from .geology_analysis_tools import GeologyStatisticsTool
from .geology_interpretation_tools import GeologyInterpretationTool
from .orchestrator import GeologyToolsOrchestrator, geology_tools_orchestrator
from .config import *

__all__ = [
    'GeologyWebSearchTool',
    'WebResearchManager',
    'GeologyStatisticsTool',
    'GeologyInterpretationTool',
    'GeologyToolsOrchestrator',
    'geology_tools_orchestrator',
    # Configuration
    'RESISTIVITY_THRESHOLDS',
    'GEOLOGICAL_FORMATIONS',
    'STATISTICAL_PARAMS',
    'WEB_SEARCH_PARAMS',
    'GEOLOGICAL_INTERPRETATION_PARAMS',
    'get_resistivity_category',
    'get_formation_info',
    'validate_api_keys'
]

__version__ = "1.0.0"
__author__ = "KIBALI Geological Analysis System"