<template>
  <div class="log-container">
    <div class="log-header">
      <span class="section-title">{{ $t('monitor.logs') }}</span>
      <el-button text size="small" @click="runnerStore.clearLogs">{{ $t('monitor.clearLog') }}</el-button>
    </div>
    <div class="log-area" ref="logEl">
      <div v-for="(line, i) in runnerStore.logs" :key="i" class="log-line">{{ line }}</div>
      <div v-if="runnerStore.logs.length === 0" class="log-empty">{{ $t('monitor.noLogsYet') }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick, onMounted, onUnmounted } from 'vue'
import { useRunnerStore } from '@/stores/runner'
import { runnerApi } from '@/api'

const runnerStore = useRunnerStore()
const logEl = ref<HTMLElement | null>(null)
let es: EventSource | null = null

function scrollBottom() {
  nextTick(() => {
    if (logEl.value) logEl.value.scrollTop = logEl.value.scrollHeight
  })
}

function connect() {
  if (es) es.close()
  es = runnerApi.connectLog(
    (d) => { runnerStore.addLog(d.line); scrollBottom() },
    (code) => {
      runnerStore.running = false
      if (code !== undefined) runnerStore.exitCode = code ?? null
      es = null
    },
  )
}

onMounted(() => {
  if (runnerStore.running) connect()
})

watch(() => runnerStore.running, (v) => {
  if (v) connect()
})

onUnmounted(() => { if (es) es.close() })
</script>

<style scoped>
.log-container { height: 100%; display: flex; flex-direction: column; }
.log-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.section-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.log-area {
  flex: 1;
  min-height: 400px;
}
.log-line {
  white-space: pre-wrap;
  word-break: break-all;
}
.log-empty {
  color: #555;
  font-style: italic;
  padding: 20px;
  text-align: center;
}
</style>
