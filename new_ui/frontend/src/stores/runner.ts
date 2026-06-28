import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface LossPoint {
  step: number
  loss: number
}

export const useRunnerStore = defineStore('runner', () => {
  const running = ref(false)
  const exitCode = ref<number | null>(null)
  const logs = ref<string[]>([])
  const command = ref('')
  const lossData = ref<LossPoint[]>([])
  let lossStep = 0

  function _pushLossFromLine(line: string) {
    const m =
      line.match(/['\"]?(?:loss|train_loss)['\"]?\s*[:=]\s*([\d.]+(?:e[+-]?\d+)?)/i) ||
      line.match(/\{[^}]*['\"]loss['\"]\s*:\s*([\d.]+(?:e[+-]?\d+)?)/i)
    if (!m) return
    const loss = parseFloat(m[1])
    if (Number.isNaN(loss)) return
    lossStep += 1
    lossData.value = [...lossData.value, { step: lossStep, loss }]
  }

  function addLog(line: string) {
    logs.value.push(line)
    _pushLossFromLine(line)
  }

  function clearLogs() {
    logs.value = []
    exitCode.value = null
    command.value = ''
    lossData.value = []
    lossStep = 0
  }

  return { running, exitCode, logs, command, lossData, addLog, clearLogs }
})
