<template>
  <div class="monitor">
    <div class="monitor-header">
      <h2>{{ $t('monitor.title') }}</h2>
      <div class="status-indicator">
        <el-tag :type="statusType" effect="dark" round>
          {{ statusText }}
        </el-tag>
      </div>
    </div>

    <el-row :gutter="16">
      <el-col :span="6">
        <RunHistory />
      </el-col>
      <el-col :span="18" class="right-col">
        <LossChart />
        <LogStream />
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRunnerStore } from '@/stores/runner'
import LogStream from '@/components/monitor/LogStream.vue'
import RunHistory from '@/components/monitor/RunHistory.vue'
import LossChart from '@/components/monitor/LossChart.vue'

const { t } = useI18n()
const runnerStore = useRunnerStore()

const statusType = computed(() => {
  if (runnerStore.running) return 'warning'
  if (runnerStore.exitCode === 0) return 'success'
  if (runnerStore.exitCode !== null) return 'danger'
  return 'info'
})

const statusText = computed(() => {
  if (runnerStore.running) return t('monitor.running')
  if (runnerStore.exitCode === 0) return t('monitor.success')
  if (runnerStore.exitCode !== null) return t('monitor.failed')
  return t('monitor.idle')
})
</script>

<style scoped>
.monitor { max-width: 1200px; margin: 0 auto; }
.monitor-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}
.monitor-header h2 { font-size: 18px; font-weight: 600; }
.right-col {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
</style>
