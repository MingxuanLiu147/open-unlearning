import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export const useAgentStore = defineStore('agent', () => {
  const messages = ref<ChatMessage[]>([])
  const streaming = ref(false)
  const currentChunk = ref('')

  function addUserMessage(content: string) {
    messages.value.push({ role: 'user', content })
  }

  function startAssistant() {
    streaming.value = true
    currentChunk.value = ''
    messages.value.push({ role: 'assistant', content: '' })
  }

  function appendChunk(text: string) {
    currentChunk.value += text
    const last = messages.value[messages.value.length - 1]
    if (last && last.role === 'assistant') {
      last.content = currentChunk.value
    }
  }

  function finishAssistant() {
    streaming.value = false
    currentChunk.value = ''
  }

  function clearHistory() {
    messages.value = []
    currentChunk.value = ''
    streaming.value = false
  }

  return { messages, streaming, currentChunk, addUserMessage, startAssistant, appendChunk, finishAssistant, clearHistory }
})
