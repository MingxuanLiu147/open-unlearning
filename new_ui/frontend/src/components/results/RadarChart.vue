<template>
  <div ref="chartEl" style="width: 100%; height: 400px;"></div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'

const props = defineProps<{
  data: Record<string, any>
  labels: string[]
}>()

const chartEl = ref<HTMLElement | null>(null)
let chart: echarts.ECharts | null = null

function render() {
  if (!chartEl.value || Object.keys(props.data).length === 0) return
  if (!chart) chart = echarts.init(chartEl.value)

  const firstEval = Object.keys(props.data)[0]
  if (!firstEval) return
  const metrics = Object.keys(props.data[firstEval])

  const series = props.labels.map(label => ({
    name: label.split('/').pop() || label,
    type: 'radar' as const,
    data: [{
      value: metrics.map(m => {
        const v = props.data[firstEval]?.[m]?.[label]
        return typeof v === 'number' ? v : 0
      }),
    }],
  }))

  chart.setOption({
    tooltip: {},
    legend: { data: series.map(s => s.name), bottom: 0, textStyle: { fontSize: 11 } },
    radar: {
      indicator: metrics.map(m => ({ name: m, max: 1 })),
      shape: 'polygon' as const,
    },
    series,
  }, true)
}

onMounted(() => { render() })
watch(() => [props.data, props.labels], render, { deep: true })
onUnmounted(() => { chart?.dispose() })
</script>
