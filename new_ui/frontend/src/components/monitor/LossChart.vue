<template>
  <div class="loss-chart">
    <div class="section-title">{{ $t('monitor.lossChart') }}</div>
    <v-chart v-if="hasData" class="chart" :option="option" autoresize />
    <div v-else class="empty">{{ $t('monitor.noLossYet') }}</div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import VChart from 'vue-echarts'
import { useRunnerStore } from '@/stores/runner'

use([CanvasRenderer, LineChart, GridComponent, TooltipComponent])

const runnerStore = useRunnerStore()

const hasData = computed(() => runnerStore.lossData.length > 0)

const option = computed(() => ({
  tooltip: { trigger: 'axis' },
  grid: { left: 48, right: 16, top: 16, bottom: 32 },
  xAxis: {
    type: 'category',
    name: 'step',
    data: runnerStore.lossData.map((d) => String(d.step)),
  },
  yAxis: { type: 'value', name: 'loss', scale: true },
  series: [
    {
      type: 'line',
      smooth: 0.2,
      showSymbol: runnerStore.lossData.length < 40,
      data: runnerStore.lossData.map((d) => d.loss),
      lineStyle: { width: 2 },
      areaStyle: { opacity: 0.06 },
    },
  ],
}))
</script>

<style scoped>
.loss-chart {
  min-height: 260px;
  margin-bottom: 12px;
}
.section-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 8px;
}
.chart {
  height: 240px;
  width: 100%;
}
.empty {
  height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  font-size: 13px;
  border: 1px dashed var(--border-color);
  border-radius: var(--radius-lg);
}
</style>
