<template>
  <div class="copilot">
    <!-- 1. Context Banner -->
    <ContextBanner />

    <!-- 2. Session Header -->
    <div class="copilot-header">
      <h3>{{ $t('copilot.title') }}</h3>
      <div class="header-actions">
        <el-dropdown
          v-if="agentStore.sessions.length > 1"
          trigger="click"
          @command="handleSessionSwitch"
          size="small"
        >
          <el-button text size="small" class="session-switch-btn">
            {{ currentSessionTitle }}
            <el-icon class="el-icon--right"><ArrowDown /></el-icon>
          </el-button>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item
                v-for="s in agentStore.sessions"
                :key="s.id"
                :command="s.id"
                :class="{ 'is-active': s.id === agentStore.currentSessionId }"
              >
                {{ s.title }}
              </el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
        <el-button text size="small" @click="handleNewSession">
          {{ $t('copilot.newChat') }}
        </el-button>
        <el-button text size="small" @click="agentStore.clearHistory">
          {{ $t('copilot.clear') }}
        </el-button>
      </div>
    </div>

    <!-- 3. Messages Area -->
    <div class="copilot-messages" ref="msgContainer">
      <WelcomePanel
        v-if="agentStore.messages.length === 0"
        @ask="handleQuickAsk"
      />
      <template v-for="(msg, i) in agentStore.messages" :key="i">
        <ChatMessage :message="msg" />
        <template v-if="msg.actions?.length">
          <ActionCard
            v-for="action in msg.actions"
            :key="action.id"
            :action="action"
            @apply="handleApply"
            @reject="handleReject"
          />
        </template>
      </template>
      <ThinkingIndicator v-if="agentStore.streaming && !agentStore.currentChunk" />
    </div>

    <!-- 4. Skill Suggestion Bar -->
    <SkillSuggestionBar
      v-if="agentStore.suggestedSkills.length > 0"
      :skills="agentStore.suggestedSkills"
      :activeSkill="agentStore.activeSkill"
      @select="agentStore.setActiveSkill"
      @clear="agentStore.setActiveSkill(null)"
    />

    <!-- 5. Input Area -->
    <div class="copilot-input">
      <div class="input-row">
        <el-input
          v-model="input"
          type="textarea"
          :autosize="{ minRows: 1, maxRows: 4 }"
          :placeholder="$t('copilot.placeholder')"
          @keydown.enter.exact.prevent="send"
          :disabled="agentStore.streaming"
          resize="none"
        />
        <el-button
          @click="agentStore.streaming ? agentStore.abortStream() : send()"
          :type="agentStore.streaming ? 'danger' : 'primary'"
          circle
          size="small"
          class="send-btn"
        >
          <el-icon :size="16">
            <component :is="agentStore.streaming ? CloseBold : Promotion" />
          </el-icon>
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, watch } from 'vue'
import { Promotion, CloseBold, ArrowDown } from '@element-plus/icons-vue'
import { useAgentStore } from '@/stores/agent'
import type { AgentAction } from '@/stores/agent'
import { agentApi } from '@/api'
import ChatMessage from './ChatMessage.vue'
import ActionCard from './ActionCard.vue'
import ContextBanner from './ContextBanner.vue'
import WelcomePanel from './WelcomePanel.vue'
import ThinkingIndicator from './ThinkingIndicator.vue'
import SkillSuggestionBar from './SkillSuggestionBar.vue'

const agentStore = useAgentStore()
const input = ref('')
const msgContainer = ref<HTMLElement | null>(null)

function scrollBottom() {
  nextTick(() => {
    if (msgContainer.value) {
      msgContainer.value.scrollTop = msgContainer.value.scrollHeight
    }
  })
}

watch(() => agentStore.messages.length, scrollBottom)
watch(() => agentStore.currentChunk, scrollBottom)

const currentSessionTitle = computed(() => {
  return agentStore.currentSession?.title || 'Chat'
})

function handleNewSession() {
  agentStore.createSession()
}

function handleSessionSwitch(sessionId: string) {
  agentStore.switchSession(sessionId)
}

function handleQuickAsk(question: string) {
  input.value = question
  send()
}

function handleApply(action: AgentAction) {
  agentStore.applyAction(action)
}

function handleReject(actionId: string) {
  agentStore.rejectAction(actionId)
}

function send() {
  const text = input.value.trim()
  if (!text || agentStore.streaming) return
  input.value = ''

  agentStore.addUserMessage(text)
  agentStore.startAssistant()

  const context = agentStore.buildContextSnapshot()

  const msgs = agentStore.messages
    .filter(m => !(m.role === 'assistant' && !m.content))
    .map(m => ({ role: m.role, content: m.content }))

  const ctrl = agentApi.streamChat(msgs, context, {
    onChunk: (chunk) => {
      agentStore.appendChunk(chunk)
      scrollBottom()
    },
    onActions: (actions) => {
      agentStore.attachActions(actions)
      scrollBottom()
    },
    onSkillSuggestions: (skills) => {
      agentStore.setSuggestedSkills(skills as any)
    },
    onDone: () => {
      agentStore.finishAssistant()
    },
    onError: (msg) => {
      agentStore.appendChunk(`\n\n[Error] ${msg}`)
      agentStore.finishAssistant()
    },
  })

  agentStore.setController(ctrl)
}
</script>

<style scoped>
.copilot {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--bg-copilot);
}

.copilot-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border-color);
}
.copilot-header h3 {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}
.header-actions {
  display: flex;
  align-items: center;
  gap: 2px;
}
.session-switch-btn {
  max-width: 100px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 12px;
}

.copilot-messages {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}

.copilot-input {
  padding: 10px 12px;
  border-top: 1px solid var(--border-color);
  background: var(--bg-copilot);
}
.input-row {
  display: flex;
  align-items: flex-end;
  gap: 8px;
}
.input-row :deep(.el-textarea__inner) {
  background: var(--bg-input);
  border-color: var(--border-color);
  border-radius: var(--radius-md);
  font-size: 13px;
  line-height: 1.5;
  padding: 8px 12px;
  resize: none;
}
.input-row :deep(.el-textarea__inner:focus) {
  border-color: var(--border-focus);
}
.send-btn {
  flex-shrink: 0;
  width: 32px;
  height: 32px;
}
</style>
