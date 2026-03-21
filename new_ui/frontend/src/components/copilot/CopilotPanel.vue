<template>
  <div class="copilot">
    <div class="copilot-header">
      <h3>{{ $t('copilot.title') }}</h3>
      <el-button text size="small" @click="agentStore.clearHistory">{{ $t('copilot.clear') }}</el-button>
    </div>

    <div class="copilot-messages" ref="msgContainer">
      <div v-if="agentStore.messages.length === 0" class="empty-hint">
        <el-icon :size="32" color="var(--text-muted)"><ChatDotRound /></el-icon>
        <p>{{ $t('copilot.placeholder') }}</p>
      </div>
      <ChatMessage
        v-for="(msg, i) in agentStore.messages"
        :key="i"
        :message="msg"
      />
    </div>

    <div class="copilot-input">
      <el-input
        v-model="input"
        :placeholder="$t('copilot.placeholder')"
        @keyup.enter="send"
        :disabled="agentStore.streaming"
        size="default"
      >
        <template #append>
          <el-button @click="send" :loading="agentStore.streaming" type="primary">
            {{ $t('copilot.send') }}
          </el-button>
        </template>
      </el-input>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, watch } from 'vue'
import { ChatDotRound } from '@element-plus/icons-vue'
import { useAgentStore } from '@/stores/agent'
import { useExperimentStore } from '@/stores/experiment'
import { useSkillsStore } from '@/stores/skills'
import { agentApi } from '@/api'
import ChatMessage from './ChatMessage.vue'

const agentStore = useAgentStore()
const experimentStore = useExperimentStore()
const skillsStore = useSkillsStore()
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

function send() {
  const text = input.value.trim()
  if (!text || agentStore.streaming) return
  input.value = ''

  agentStore.addUserMessage(text)
  agentStore.startAssistant()

  const context = {
    mode: experimentStore.mode,
    model: experimentStore.selectedModel,
    trainer: experimentStore.selectedTrainer,
    datasets: experimentStore.selectedDatasets,
    skills: skillsStore.skills.map((s: any) => s.name),
  }

  const msgs = agentStore.messages
    .filter(m => !(m.role === 'assistant' && !m.content))
    .map(m => ({ role: m.role, content: m.content }))

  agentApi.streamChat(
    msgs,
    context,
    (chunk) => { agentStore.appendChunk(chunk); scrollBottom() },
    () => agentStore.finishAssistant(),
  )
}
</script>

<style scoped>
.copilot {
  display: flex;
  flex-direction: column;
  height: 100%;
}
.copilot-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-color);
}
.copilot-header h3 {
  font-size: 14px;
  font-weight: 600;
}
.copilot-messages {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}
.empty-hint {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 40px 20px;
  color: var(--text-muted);
  text-align: center;
  font-size: 13px;
}
.copilot-input {
  padding: 12px;
  border-top: 1px solid var(--border-color);
}
</style>
