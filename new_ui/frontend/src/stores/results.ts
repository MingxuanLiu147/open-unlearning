import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useResultsStore = defineStore('results', () => {
  const runs = ref<any[]>([])
  const selectedLabels = ref<string[]>([])
  const compareData = ref<Record<string, any>>({})

  return { runs, selectedLabels, compareData }
})
