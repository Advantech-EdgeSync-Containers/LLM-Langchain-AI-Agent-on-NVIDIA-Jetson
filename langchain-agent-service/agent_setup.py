# agent_setup.py
import codecs
import json
import os
import re
from typing import List, Tuple, Any, Optional
from pydantic import PrivateAttr
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import BaseSingleActionAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import tools, BaseTool

class MyLLMDecisionAgent(BaseSingleActionAgent):
    _tools: List[BaseTool] = PrivateAttr()
    _llm: Any = PrivateAttr()
    _prompt: Any = PrivateAttr()
    _output_parser: Any = PrivateAttr()

    def __init__(self, tools: List[BaseTool], llm: Any):
        super().__init__()
        object.__setattr__(self, "_tools", tools)
        object.__setattr__(self, "_llm", llm)

        system_prompt = """
        You are a smart assistant capable of both hardware control tasks and general natural language conversations.

        For **hardware control and monitoring tasks**, you are connected to a set of tools:
        - device_info_tool → Retrieves motherboard and BIOS details (manufacturer, model, BIOS version, etc.).
        - device_voltage_tool → Returns voltage readings from onboard voltage sensors.
        - device_temperature_tool → Returns temperature readings from onboard temperature sensors.
        - device_fans_tool → Returns real-time fan speed (RPM) readings from onboard sensors.
        - gpio_pins_overview → Provides an overview of GPIO pins, including their direction and current logic level.
        - gpio_set_tool → Sets the output level of a GPIO pin. Expects input in the format: pin=PIN_NAME, level=LEVEL (e.g., pin=GPIO1, level=HIGH).
        - gpio_read_tool → Reads the current logic level of a specific GPIO pin. Expects input in the format: pin=PIN_NAME (e.g., pin=GPIO1).

        **Use these tools only if the user clearly and explicitly asks for a hardware-related action or measurement.** If the user's request is general, common sense-based, or not clearly hardware-specific, respond using your natural language knowledge.

        **For hardware tool usage**, respond only with:
        - For **gpio_set_tool** → tool_name: gpio_set_tool, tool_pin: PIN_NAME, tool_level: LEVEL.
        - For other tools → tool_name only.

        **For general or creative tasks** (e.g., stories, questions, puzzles, trivia, general knowledge), respond naturally in complete sentences, without using any tools.

        Examples:

        User: What is the board info?  
        → tool_name: device_info_tool

        User: Show me voltages of the system  
        → tool_name: device_voltage_tool

        User: What’s the temperature of the device?  
        → tool_name: device_temperature_tool

        User: What’s the fan speed of the device?  
        → tool_name: device_fans_tool

        User: Show me the GPIO overview  
        → tool_name: gpio_pins_overview

        User: Set GPIO pin GPIO5 to low  
        → tool_name: gpio_set_tool  
        → tool_pin: GPIO5  
        → tool_level: LOW

        User: Read GPIO pin GPIO3 level  
        → tool_name: gpio_read_tool  
        → tool_pin: GPIO3

        User: Write a short story on a haunted house  
        → In the dead of night, the old mansion creaked and groaned as though whispering secrets to the darkness. Windows rattled in the wind, and the flicker of candlelight revealed ghostly figures in every shadow...

        User: Who is the President of France?  
        → Emmanuel Macron

        User: What comes next in the sequence SCD, TEF, UGH?  
        → VJI
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt.strip()),
            ("human", "{question}")
        ])

        object.__setattr__(self, "_prompt", prompt)
        object.__setattr__(self, "_output_parser", StrOutputParser())
        object.__setattr__(self, "_chain", self._prompt | self._llm | self._output_parser)

    def extract_json_key(self, code_block: str, key: str) -> Optional[str]:
        """
        Extract the value of a specified key from a markdown-wrapped JSON code block.
        """
        try:
            # Remove triple backticks and optional language specifier like ```json
            cleaned = re.sub(r"```[a-zA-Z]*", "", code_block).replace("```", "").strip()
            parsed = json.loads(cleaned)
            return parsed.get(key)
        except (json.JSONDecodeError, TypeError):
            return None

    def remove_think_block(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    def replace_tool_with_output(self, text: str, output: str) -> str:
        return re.sub(r"(?i)^tool_name:\s*\w+\s*$", output, text, flags=re.MULTILINE)

    def extract_tool(self, text: str) -> dict:
        tool_names = [tool.name.lower() for tool in tools]
        text = codecs.decode(text, 'unicode_escape')

        # Match the tool name
        pattern = r"\b(" + "|".join(re.escape(tool) for tool in tool_names) + r")\b"
        tool_match = re.search(pattern, text)

        if not tool_match:
            return {}

        tool_name = tool_match.group(1).strip()

        result = {"tool_name": tool_name}

        if tool_name in ["gpio_set_tool" ,'gpio_read_tool']:
            # Extract tool_pin
            pin_match = re.search(r"tool_pin:\s*([\w\d]+)", text, re.IGNORECASE)
            level_match = re.search(r"tool_level:\s*([\w\d]+)", text, re.IGNORECASE)

            if pin_match:
                result["tool_pin"] = pin_match.group(1).strip()
            if level_match:
                result["tool_level"] = level_match.group(1).strip().upper()

        return result

    async def ainvoke_tool_from_text(self, text: str, original_input: str,callback_handler=None) -> str:

        tool_name, tool_input = self.extract_tool_and_input(text)
        tool_lookup = {tool.name.lower(): tool for tool in self._tools}

        print(f"tool: {str(tool_name)}")
        print(f"tool_input: {str(tool_input)}")
        if not tool_name or tool_name == "none" or tool_name not in tool_lookup :
            if callback_handler:
                # Stream response from LLM with callback
                self._llm.callbacks = [callback_handler]
                await self._llm.ainvoke(original_input)
                return None  # signal: already streamed
            else:
                result = await self._llm.ainvoke(original_input)
                return self.remove_think_block(result)

        if tool_name in tool_lookup:
            tool_result = tool_lookup[tool_name].run({"input": tool_input})
            return tool_result

        return f"Unknown tool: {tool_name}"

    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any) -> AgentFinish:
        raise NotImplementedError("Use aplan() with streaming")

    async def aplan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any) -> AgentFinish:
        prompt = kwargs["input"]
        streamed_text = await self._chain.ainvoke({"question": prompt})
        final_result = await self.ainvoke_tool_from_text(streamed_text, prompt)
        return AgentFinish(return_values={"output": final_result}, log="Streaming + tool resolved")

    @property
    def input_keys(self) -> List[str]:
        return ["input"]