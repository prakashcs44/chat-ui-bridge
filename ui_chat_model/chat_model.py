from typing import Any, List, Optional
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from playwright.async_api import async_playwright
import asyncio

class ChatUIBridge(BaseChatModel):
    """
    A custom LangChain chat model that generates responses by 
    automating a UI interaction via Playwright.
    """
    
    url: str = "http://127.0.0.1:5500/frontend/index.html"
    model_name: str = "llama-3.1-8b-instant"
    headless: bool = True
    timeout_ms: int = 30000

    @property
    def _llm_type(self) -> str:
        return "ui-bridge-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous implementation using asyncio.run"""
        return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        The core logic: Automate the browser to get the LLM response.
        """
        # 1. Format the full conversation history
        # This ensures the AI remembers previous context and follows System Instructions
        history = ""
        for msg in messages:
            prefix = "Human: " if msg.type == "human" else "System: " if msg.type == "system" else "AI: "
            history += f"{prefix}{msg.content}\n"
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            page = await browser.new_page()
            
            try:
                await page.goto(self.url)
                
                # 2. Select the Model
                await page.select_option("#modelSelect", self.model_name)
                
                # 3. Fill input (with full history) & Click Submit
                await page.fill("#userInput", history.strip())
                await page.click("#submitBtn")
                
                # 4. Wait for result
                await page.wait_for_selector("#result-container", state="visible", timeout=self.timeout_ms)
                
                # 5. Extract PURE text
                content = await page.inner_text("#output")
                
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
                
            except Exception as e:
                raise RuntimeError(f"UI Automation failed: {str(e)}")
            finally:
                await browser.close()

    @property
    def _identifying_params(self) -> dict:
        return {"url": self.url, "model_name": self.model_name, "headless": self.headless}
