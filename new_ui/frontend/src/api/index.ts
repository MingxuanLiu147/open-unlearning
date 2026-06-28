const BASE = ''

async function get<T = any>(url: string): Promise<T> {
  const res = await fetch(BASE + url)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

async function post<T = any>(url: string, body?: any): Promise<T> {
  const res = await fetch(BASE + url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

async function put<T = any>(url: string, body?: any): Promise<T> {
  const res = await fetch(BASE + url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

async function del<T = any>(url: string): Promise<T> {
  const res = await fetch(BASE + url, { method: 'DELETE' })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const configApi = {
  getModels: () => get('/api/config/models'),
  getTrainers: (mode?: string) => get(`/api/config/trainers${mode ? `?mode=${mode}` : ''}`),
  getDatasets: (mode?: string) => get(`/api/config/datasets${mode ? `?mode=${mode}` : ''}`),
  getEvals: (mode?: string) => get(`/api/config/evals${mode ? `?mode=${mode}` : ''}`),
  getExperiments: (mode?: string) => get(`/api/config/experiments${mode ? `?mode=${mode}` : ''}`),
  getTrainerParams: (name: string) => get(`/api/config/trainer-params?name=${name}`),
  importConfig: (data: any) => post('/api/config/import', data),
  exportConfig: (data: any) => post('/api/config/export', data),
  createModel: (data: any) => post('/api/config/create-model', data),
  getMethodCatalog: (scenario?: string) =>
    get(`/api/config/method-catalog${scenario ? `?scenario=${encodeURIComponent(scenario)}` : ''}`),
}

export const runnerApi = {
  start: (body: any) => post('/api/run/start', body),
  stop: () => post('/api/run/stop'),
  status: () => get('/api/run/status'),
  connectLog: (
    onData: (d: any) => void,
    onDone?: (exitCode: number | null | undefined) => void,
  ) => {
    const es = new EventSource('/api/run/log')
    es.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.done) {
        es.close()
        onDone?.(data.exit_code)
      } else {
        onData(data)
      }
    }
    es.onerror = () => {
      es.close()
      onDone?.(undefined)
    }
    return es
  },
}

export const resultsApi = {
  list: () => get('/api/results/list'),
  get: (id: string) => get(`/api/results/${encodeURIComponent(id)}`),
  compare: (labels: string[]) => post('/api/results/compare', { labels }),
  behaviorCompare: (labels: string[], question?: string) =>
    post('/api/results/behavior-compare', { labels, question }),
}

export const skillsApi = {
  list: () => get('/api/skills'),
  get: (id: string) => get(`/api/skills/${id}`),
  create: (data: any) => post('/api/skills', data),
  update: (id: string, data: any) => put(`/api/skills/${id}`, data),
  delete: (id: string) => del(`/api/skills/${id}`),
  run: (id: string) => post(`/api/skills/${id}/run`),
}

export interface StreamChatCallbacks {
  onChunk: (text: string) => void
  onActions: (actions: any[]) => void
  onSkillSuggestions: (skills: string[]) => void
  onDone: () => void
  onError: (msg: string) => void
}

export const agentApi = {
  getConfig: () => get('/api/agent/config'),
  putConfig: (data: any) => put('/api/agent/config', data),
  test: (data: any) => post('/api/agent/test', data),
  getSkills: () => get('/api/agent/skills'),
  streamChat: (
    messages: any[],
    context: any,
    callbacks: StreamChatCallbacks,
  ) => {
    const ctrl = new AbortController()
    fetch('/api/agent/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages, context }),
      signal: ctrl.signal,
    }).then(async (res) => {
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const parts = buf.split('\n\n')
        buf = parts.pop() || ''
        for (const part of parts) {
          if (!part.startsWith('data: ')) continue
          try {
            const d = JSON.parse(part.slice(6))
            switch (d.type) {
              case 'message_delta':     callbacks.onChunk(d.content); break
              case 'actions':           callbacks.onActions(d.actions); break
              case 'skill_suggestions': callbacks.onSkillSuggestions(d.skills); break
              case 'error':             callbacks.onError(d.message); break
              case 'done':              callbacks.onDone(); return
            }
          } catch { /* skip malformed frames */ }
        }
      }
      callbacks.onDone()
    }).catch(() => callbacks.onDone())
    return ctrl
  },
}

export const dataApi = {
  parse: (text: string) => post('/api/data/parse', { text }),
  validate: (records: any[]) => post('/api/data/validate', { records }),
  validatePath: (path: string, mode: string) =>
    post('/api/data/validate', { path, mode }),
  uploadWithValidate: async (file: File, mode: string, purpose: string) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('mode', mode)
    fd.append('purpose', purpose)
    const res = await fetch(BASE + '/api/data/upload', { method: 'POST', body: fd })
    return res.json()
  },
}
