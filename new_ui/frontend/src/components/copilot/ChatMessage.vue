<template>
  <div class="chat-msg" :class="message.role">
    <div class="msg-avatar" :class="'avatar-' + message.role">
      <span v-if="message.role === 'user'">U</span>
      <svg v-else viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
        <path d="M12 2a2 2 0 012 2c0 .74-.4 1.39-1 1.73v.27h1a7 7 0 017 7h-2a5 5 0 00-5-5h-4a5 5 0 00-5 5H3a7 7 0 017-7h1v-.27A2 2 0 0112 2zm-4 16a2 2 0 114 0 2 2 0 01-4 0zm6 0a2 2 0 114 0 2 2 0 01-4 0z"/>
      </svg>
    </div>
    <div class="msg-body" v-html="rendered"></div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { marked } from 'marked'
import type { ChatMessage } from '@/stores/agent'

const props = defineProps<{ message: ChatMessage }>()

const rendered = computed(() => {
  try {
    return marked.parse(props.message.content || '...', { async: false })
  } catch {
    return props.message.content
  }
})
</script>

<style scoped>
.chat-msg {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}
.chat-msg.user { flex-direction: row-reverse; }
.msg-avatar {
  flex-shrink: 0;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 700;
  color: #fff;
}
.avatar-user {
  background: var(--accent-primary);
}
.avatar-assistant {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.msg-body {
  max-width: 85%;
  padding: 8px 12px;
  border-radius: var(--radius-md);
  font-size: 13px;
  line-height: 1.6;
  word-break: break-word;
}
.chat-msg.user .msg-body {
  background: var(--accent-primary);
  color: white;
  border-bottom-right-radius: 2px;
}
.chat-msg.assistant .msg-body {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-bottom-left-radius: 2px;
}
.msg-body :deep(pre) {
  background: var(--bg-code);
  padding: 8px;
  border-radius: 4px;
  overflow-x: auto;
  font-size: 12px;
  margin: 6px 0;
}
.msg-body :deep(code) {
  font-family: var(--font-mono);
  font-size: 12px;
}
.msg-body :deep(p) { margin: 4px 0; }
</style>
