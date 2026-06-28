import { defineStore } from 'pinia'
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useExperimentStore } from './experiment'
import { useSkillsStore } from './skills'
import { useRunnerStore } from './runner'
import { useResultsStore } from './results'

// ─── Types ───

export type ActionType = 'apply_config' | 'navigate_view'
export type ActionStatus = 'pending' | 'applied' | 'rejected'
export type SkillId =
  | 'config_wizard' | 'param_tuner' | 'skill_recommender' | 'edit_config_guide'
  | 'dataset_guide' | 'forget_retain_advisor'
  | 'result_analyzer' | 'next_step_advisor'
  | 'concept_explainer' | 'method_comparator'

export interface AgentAction {
  id: string
  type: ActionType
  preview: string
  payload: Record<string, any>
  status: ActionStatus
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  actions?: AgentAction[]
  timestamp: number
}

export interface AgentSession {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: number
  updatedAt: number
}

export interface AgentContextSnapshot {
  route: string
  mode: string
  model: string
  trainer: string
  datasets: Record<string, string>
  eval: string
  experiment: string
  params: Record<string, any>
  skills: string[]
  activeSkill: SkillId | null
  runner?: { running: boolean; exitCode: number | null }
  results?: { selectedLabels: string[]; compareData: Record<string, any> }
}

const STORAGE_KEY = 'ks-copilot-sessions'
const CURRENT_SESSION_KEY = 'ks-copilot-current'

function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7)
}

// ─── Store ───

export const useAgentStore = defineStore('agent', () => {
  const sessions = ref<AgentSession[]>([])
  const currentSessionId = ref<string>('')
  const streaming = ref(false)
  const currentChunk = ref('')
  const activeSkill = ref<SkillId | null>(null)
  const suggestedSkills = ref<SkillId[]>([])
  const controller = ref<AbortController | null>(null)

  const currentSession = computed(() =>
    sessions.value.find(s => s.id === currentSessionId.value)
  )

  const messages = computed(() => currentSession.value?.messages ?? [])

  // ─── Session management ───

  function createSession(): string {
    const id = generateId()
    const session: AgentSession = {
      id,
      title: `Chat ${sessions.value.length + 1}`,
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    }
    sessions.value.push(session)
    currentSessionId.value = id
    persistToLocalStorage()
    return id
  }

  function ensureSession() {
    if (!currentSession.value) {
      createSession()
    }
  }

  function switchSession(id: string) {
    if (sessions.value.some(s => s.id === id)) {
      currentSessionId.value = id
      localStorage.setItem(CURRENT_SESSION_KEY, id)
    }
  }

  function deleteSession(id: string) {
    sessions.value = sessions.value.filter(s => s.id !== id)
    if (currentSessionId.value === id) {
      currentSessionId.value = sessions.value[0]?.id ?? ''
      if (!currentSessionId.value) createSession()
    }
    persistToLocalStorage()
  }

  // ─── Message operations ───

  function addUserMessage(content: string) {
    ensureSession()
    const session = currentSession.value!
    session.messages.push({ role: 'user', content, timestamp: Date.now() })
    session.updatedAt = Date.now()
    if (session.messages.length === 1) {
      session.title = content.slice(0, 30) || 'New Chat'
    }
    persistToLocalStorage()
  }

  function startAssistant() {
    ensureSession()
    streaming.value = true
    currentChunk.value = ''
    currentSession.value!.messages.push({
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
    })
  }

  function appendChunk(text: string) {
    currentChunk.value += text
    const session = currentSession.value
    if (!session) return
    const last = session.messages[session.messages.length - 1]
    if (last && last.role === 'assistant') {
      last.content = currentChunk.value
    }
  }

  function attachActions(actions: AgentAction[]) {
    const session = currentSession.value
    if (!session) return
    const last = session.messages[session.messages.length - 1]
    if (last && last.role === 'assistant') {
      last.actions = actions
    }
    persistToLocalStorage()
  }

  function finishAssistant() {
    streaming.value = false
    currentChunk.value = ''
    if (currentSession.value) {
      currentSession.value.updatedAt = Date.now()
    }
    persistToLocalStorage()
  }

  // ─── Action handling ───

  function applyAction(action: AgentAction) {
    action.status = 'applied'

    if (action.type === 'apply_config') {
      const experimentStore = useExperimentStore()
      experimentStore.applyConfig(action.payload)
    } else if (action.type === 'navigate_view') {
      const router = useRouter()
      const view = action.payload.view
      if (view) router.push(`/${view}`)
    }

    persistToLocalStorage()
  }

  function rejectAction(actionId: string) {
    const session = currentSession.value
    if (!session) return
    for (const msg of session.messages) {
      if (!msg.actions) continue
      const act = msg.actions.find(a => a.id === actionId)
      if (act) {
        act.status = 'rejected'
        break
      }
    }
    persistToLocalStorage()
  }

  // ─── Skill management ───

  function setActiveSkill(skill: SkillId | null) {
    activeSkill.value = skill
  }

  function setSuggestedSkills(skills: SkillId[]) {
    suggestedSkills.value = skills
  }

  // ─── Context snapshot ───

  function buildContextSnapshot(): AgentContextSnapshot {
    const experimentStore = useExperimentStore()
    const skillsStore = useSkillsStore()
    const runnerStore = useRunnerStore()
    const resultsStore = useResultsStore()

    const route = window.location.pathname.replace('/', '') || 'workshop'

    return {
      route,
      mode: experimentStore.mode,
      model: experimentStore.selectedModel,
      trainer: experimentStore.selectedTrainer,
      datasets: experimentStore.selectedDatasets,
      eval: experimentStore.selectedEval,
      experiment: experimentStore.selectedExperiment,
      params: experimentStore.params,
      skills: skillsStore.skills.map((s: any) => s.name || s.id),
      activeSkill: activeSkill.value,
      runner: {
        running: runnerStore.running,
        exitCode: runnerStore.exitCode,
      },
      results: {
        selectedLabels: resultsStore.selectedLabels,
        compareData: resultsStore.compareData,
      },
    }
  }

  // ─── Persistence ───

  function persistToLocalStorage() {
    try {
      const data = sessions.value.map(s => ({
        ...s,
        messages: s.messages.slice(-100),
      }))
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
      localStorage.setItem(CURRENT_SESSION_KEY, currentSessionId.value)
    } catch {
      // quota exceeded — silently skip
    }
  }

  function restoreFromLocalStorage() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (raw) {
        const data = JSON.parse(raw) as AgentSession[]
        sessions.value = data
      }
      const savedId = localStorage.getItem(CURRENT_SESSION_KEY)
      if (savedId && sessions.value.some(s => s.id === savedId)) {
        currentSessionId.value = savedId
      }
    } catch {
      // corrupted data — start fresh
    }
    if (!sessions.value.length) {
      createSession()
    } else if (!currentSessionId.value) {
      currentSessionId.value = sessions.value[0].id
    }
  }

  function clearHistory() {
    const session = currentSession.value
    if (session) {
      session.messages = []
      session.updatedAt = Date.now()
    }
    currentChunk.value = ''
    streaming.value = false
    persistToLocalStorage()
  }

  // ─── Abort ───

  function setController(ctrl: AbortController | null) {
    controller.value = ctrl
  }

  function abortStream() {
    controller.value?.abort()
    controller.value = null
    finishAssistant()
  }

  // Restore on creation
  restoreFromLocalStorage()

  return {
    sessions,
    currentSessionId,
    currentSession,
    messages,
    streaming,
    currentChunk,
    activeSkill,
    suggestedSkills,
    controller,

    createSession,
    ensureSession,
    switchSession,
    deleteSession,
    addUserMessage,
    startAssistant,
    appendChunk,
    attachActions,
    finishAssistant,
    applyAction,
    rejectAction,
    setActiveSkill,
    setSuggestedSkills,
    buildContextSnapshot,
    persistToLocalStorage,
    restoreFromLocalStorage,
    clearHistory,
    setController,
    abortStream,
  }
})
