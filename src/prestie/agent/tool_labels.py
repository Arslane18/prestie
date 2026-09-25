"""Player-facing (French) labels for tool calls, shared by the CLI and the API."""

from prestie.agent.agent import ToolCall
from prestie.agent.character_tool import CHARACTER_STATE_TOOL_NAME
from prestie.agent.quest_tool import QUEST_DETAILS_TOOL_NAME


def tool_call_label(call: ToolCall) -> str:
    if call.name == CHARACTER_STATE_TOOL_NAME:
        return "[personnage] lecture de l'état exporté par l'addon"
    if call.name == QUEST_DETAILS_TOOL_NAME:
        return f"[quête] détails de la quête {call.input.get('quest_id', '?')}"
    filters = [call.input.get(key) for key in ("spec", "content_type")]
    scope = "".join(f" [{value}]" for value in filters if value)
    return f"[recherche] {call.input.get('query', '?')}{scope}"
