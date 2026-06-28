<template>
  <div class="chat-msg" :class="message.role">
    <div class="msg-avatar" :class="'avatar-' + message.role">
      <span v-if="message.role === 'user'">U</span>
      <svg v-else viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
        <path d="M12 2a2 2 0 012 2c0 .74-.4 1.39-1 1.73v.27h1a7 7 0 017 7h-2a5 5 0 00-5-5h-4a5 5 0 00-5 5H3a7 7 0 017-7h1v-.27A2 2 0 0112 2zm-4 16a2 2 0 114 0 2 2 0 01-4 0zm6 0a2 2 0 114 0 2 2 0 01-4 0z"/>
      </svg>
    </div>
    <div class="msg-content-wrap">
      <div class="msg-body" v-html="rendered"></div>
      <div class="msg-actions" v-if="message.role === 'assistant' && message.content">
        <button class="msg-action-btn" @click="handleCopy" :title="$t('copilot.copySuccess')">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2"/>
            <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
          </svg>
          <span v-if="copied" class="copied-text">{{ $t('copilot.copySuccess') }}</span>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import hljs from 'highlight.js/lib/core'
import python from 'highlight.js/lib/languages/python'
import json from 'highlight.js/lib/languages/json'
import bash from 'highlight.js/lib/languages/bash'
import yaml from 'highlight.js/lib/languages/yaml'
import 'highlight.js/styles/github-dark-dimmed.css'
import type { ChatMessage } from '@/stores/agent'

hljs.registerLanguage('python', python)
hljs.registerLanguage('json', json)
hljs.registerLanguage('bash', bash)
hljs.registerLanguage('yaml', yaml)

const renderer = new marked.Renderer()
renderer.code = ({ text, lang }: { text: string; lang?: string }) => {
  const language = lang && hljs.getLanguage(lang) ? lang : 'plaintext'
  let highlighted: string
  try {
    highlighted = language !== 'plaintext'
      ? hljs.highlight(text, { language }).value
      : text.replace(/</g, '&lt;').replace(/>/g, '&gt;')
  } catch {
    highlighted = text.replace(/</g, '&lt;').replace(/>/g, '&gt;')
  }
  return `<pre class="hljs-block"><code class="hljs language-${language}">${highlighted}</code></pre>`
}

marked.setOptions({ renderer })

const props = defineProps<{ message: ChatMessage }>()

const copied = ref(false)

function stripActionBlock(text: string): string {
  return text.replace(/```json\s*\n?\s*\{"actions"\s*:[\s\S]*?\}\s*\n?\s*```/g, '').trim()
}

const rendered = computed(() => {
  try {
    const content = props.message.content || '...'
    const cleaned = props.message.role === 'assistant' ? stripActionBlock(content) : content
    const raw = marked.parse(cleaned, { async: false }) as string
    return DOMPurify.sanitize(raw)
  } catch {
    return DOMPurify.sanitize(props.message.content)
  }
})

function handleCopy() {
  navigator.clipboard.writeText(props.message.content).then(() => {
    copied.value = true
    setTimeout(() => { copied.value = false }, 1500)
  })
}
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
.msg-content-wrap {
  max-width: 85%;
  min-width: 0;
}
.msg-body {
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
  border-left: 2px solid #667eea;
  border-bottom-left-radius: 2px;
}
.msg-body :deep(.hljs-block) {
  background: #22272e;
  padding: 10px 12px;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 12px;
  margin: 8px 0;
}
.msg-body :deep(.hljs-block code) {
  font-family: var(--font-mono);
  font-size: 12px;
}
.msg-body :deep(code) {
  font-family: var(--font-mono);
  font-size: 12px;
  background: var(--bg-code);
  padding: 1px 4px;
  border-radius: 3px;
}
.msg-body :deep(p) { margin: 4px 0; }
.msg-body :deep(ul), .msg-body :deep(ol) {
  padding-left: 18px;
  margin: 4px 0;
}

.msg-actions {
  display: flex;
  gap: 4px;
  margin-top: 4px;
  opacity: 0;
  transition: opacity var(--transition-fast);
}
.chat-msg:hover .msg-actions {
  opacity: 1;
}
.msg-action-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 11px;
  transition: all var(--transition-fast);
}
.msg-action-btn:hover {
  background: var(--bg-card-hover);
  color: var(--text-secondary);
}
.copied-text {
  color: var(--accent-success);
  font-size: 11px;
}
</style>
