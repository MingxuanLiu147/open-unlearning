import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useRunnerStore = defineStore('runner', () => {
  const running = ref(false)
  const exitCode = ref<number | null>(null)
  const logs = ref<string[]>([])
  const command = ref('')

  function addLog(line: string) {
    logs.value.push(line)
  }

  function clearLogs() {
    logs.value = []
    exitCode.value = null
    command.value = ''
  }

  return { running, exitCode, logs, command, addLog, clearLogs }
})
