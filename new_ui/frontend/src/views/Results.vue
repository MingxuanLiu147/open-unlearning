<template>
  <div class="results-view">
    <h2>{{ $t('results.title') }}</h2>

    <el-row :gutter="16">
      <el-col :span="8">
        <div class="section-title">{{ $t('results.selectRuns') }}</div>
        <el-checkbox-group v-model="selectedLabels" class="run-list">
          <el-checkbox
            v-for="r in runs"
            :key="r.label"
            :label="r.label"
            :value="r.label"
            class="run-checkbox"
          >
            <span class="mode-badge" :class="r.mode">{{ r.mode }}</span>
            {{ r.task_name }} @ {{ r.checkpoint }}
          </el-checkbox>
        </el-checkbox-group>
        <el-button type="primary" size="small" @click="compare" :disabled="selectedLabels.length < 2" style="margin-top:8px;">
          {{ $t('results.compare') }}
        </el-button>
      </el-col>

      <el-col :span="16">
        <el-tabs>
          <el-tab-pane :label="$t('results.metrics')">
            <MetricsTable :data="compareData" :labels="selectedLabels" />
          </el-tab-pane>
          <el-tab-pane :label="$t('results.radar')">
            <RadarChart :data="compareData" :labels="selectedLabels" />
          </el-tab-pane>
        </el-tabs>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { resultsApi } from '@/api'
import MetricsTable from '@/components/results/MetricsTable.vue'
import RadarChart from '@/components/results/RadarChart.vue'

const runs = ref<any[]>([])
const selectedLabels = ref<string[]>([])
const compareData = ref<Record<string, any>>({})

onMounted(async () => {
  runs.value = await resultsApi.list()
})

async function compare() {
  if (selectedLabels.value.length < 2) return
  compareData.value = await resultsApi.compare(selectedLabels.value)
}
</script>

<style scoped>
.results-view { max-width: 1200px; margin: 0 auto; }
h2 { font-size: 18px; font-weight: 600; margin-bottom: 16px; }
.section-title {
  font-size: 13px; font-weight: 600; color: var(--text-secondary);
  margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;
}
.run-list { display: flex; flex-direction: column; gap: 4px; }
.run-checkbox { margin-bottom: 2px; }
</style>
