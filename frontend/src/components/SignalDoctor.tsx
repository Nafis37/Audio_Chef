/**
 * Signal Doctor: column 3's checklist of what is wrong with the open file, and a
 * "Fix automatically" button that appends the cards that fix it.
 *
 * Six checks: clipping, uneven volume, fan / AC noise, other background noise,
 * low-frequency rumble and long silences (3 s or more).  Every measurement and threshold
 * lives in backend/app/dsp/doctor.py; this component only draws the verdicts.  The fix is
 * NOT a hidden process: it becomes ordinary recipe cards (high-pass, Noise Remover,
 * Silence Remover, Voice Leveler) that can be tweaked or bypassed like any other, and
 * Auto-Bake plays the result.
 *
 * Two views of the same checks:
 *   Original  -- the upload as decoded, fetched once per file
 *   Processed -- the latest bake, re-fetched after every bake while it is selected, so
 *                "did the fix work?" is answered by the same checklist turning green.
 */

import { AlertTriangle, CheckCircle2, Stethoscope, Wand2, XCircle } from 'lucide-react'
import { memo, useEffect, useState } from 'react'

import { fetchDiagnosis } from '../api'
import type { DoctorReport, WireStep } from '../types'

interface Props {
  /** The open tab's upload; null for a tab with no file of its own. */
  fileId: string | null
  /** The latest bake of the open tab, if any. */
  bakeId: string | null
  onFix: (steps: WireStep[]) => void
}

type View = 'original' | 'processed'

const ICON = {
  ok: <CheckCircle2 className="size-4 shrink-0 text-[var(--chef-accent-strong)]" />,
  warn: <AlertTriangle className="size-4 shrink-0 text-amber-500" />,
  bad: <XCircle className="size-4 shrink-0 text-red-500" />,
}

/**
 * Fetches one diagnosis whenever its id changes; aborts a superseded request.  The result
 * is stored WITH the id it answers, so a stale answer is simply not shown -- no reset
 * inside the effect needed.
 */
function useDiagnosis(kind: 'file' | 'bake', id: string | null) {
  const [result, setResult] = useState<{
    id: string
    report: DoctorReport | null
    error: string | null
  } | null>(null)
  useEffect(() => {
    if (!id) return
    const controller = new AbortController()
    fetchDiagnosis(kind, id, controller.signal)
      .then((report) => setResult({ id, report, error: null }))
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === 'AbortError') return
        setResult({ id, report: null, error: reason instanceof Error ? reason.message : String(reason) })
      })
    return () => controller.abort()
  }, [kind, id])
  const current = result && result.id === id ? result : null
  return { report: current?.report ?? null, error: current?.error ?? null }
}

function SignalDoctorImpl({ fileId, bakeId, onFix }: Props) {
  // Remembered per file, so opening another tab starts on ITS original checklist.
  const [picked, setPicked] = useState<{ fileId: string | null; view: View }>({
    fileId,
    view: 'original',
  })
  const view: View = picked.fileId === fileId ? picked.view : 'original'
  const setView = (next: View) => setPicked({ fileId, view: next })
  const original = useDiagnosis('file', fileId)
  // Only diagnose bakes while someone is looking: it is a whole-file FFT per bake.
  const processed = useDiagnosis('bake', view === 'processed' ? bakeId : null)
  const { report, error } = view === 'original' ? original : processed

  // Every check is listed, worst first, so problems sit on top and passes stay visible.
  const rank = { bad: 0, warn: 1, ok: 2 }
  const rows = [...(report?.findings ?? [])].sort((a, b) => rank[a.status] - rank[b.status])
  // The backend prescribes for 'bad' findings only; 'warn' ones are minor.
  const findings = original.report?.findings ?? []
  const problems = findings.filter((f) => f.status === 'bad').length
  const minor = findings.filter((f) => f.status === 'warn').length
  const prescription = original.report?.fix ?? []

  return (
    <section className="rounded-lg border border-[var(--chef-border)] bg-[var(--chef-panel)]">
      <header className="flex items-center gap-3 border-b border-[var(--chef-border)] px-4 py-2.5">
        <Stethoscope className="size-4 text-[var(--chef-accent-strong)]" />
        <h2 className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--chef-muted)]">
          Signal Doctor
        </h2>
        <div className="ml-auto flex rounded-md border border-[var(--chef-border)] bg-[var(--chef-inset)] p-0.5">
          {(['original', 'processed'] as const).map((side) => (
            <button
              key={side}
              type="button"
              disabled={side === 'processed' && !bakeId}
              onClick={() => setView(side)}
              className={`rounded px-2.5 py-0.5 text-[11px] transition disabled:opacity-30 ${
                view === side
                  ? 'bg-[var(--chef-accent-strong)] font-medium text-white'
                  : 'text-[var(--chef-muted)] hover:text-[var(--chef-text)]'
              }`}
            >
              {side === 'original' ? 'Original' : 'Processed'}
            </button>
          ))}
        </div>
      </header>

      <div className="px-4 pb-4 pt-3">
        {error ? (
          <p className="text-xs text-red-600">{error}</p>
        ) : !report ? (
          <p className="text-xs text-[var(--chef-muted)]">
            {fileId ? 'Examining…' : 'Open a file tab to examine it.'}
          </p>
        ) : (
          <ul className="space-y-1.5">
            {rows.map((finding) => (
              <li key={finding.id} className="flex items-start gap-2 text-xs" title={finding.detail}>
                {ICON[finding.status]}
                <span className={finding.status === 'ok' ? '' : 'font-medium'}>
                  {finding.title}
                  {finding.status === 'warn' && (
                    <span className="ml-1.5 font-normal text-[var(--chef-muted)]">minor</span>
                  )}
                </span>
                <span className="ml-auto whitespace-nowrap pl-2 font-mono text-[11px] text-[var(--chef-muted)]">
                  {finding.value}
                </span>
              </li>
            ))}
          </ul>
        )}

        {view === 'original' && original.report && (
          <div className="mt-3 flex items-center gap-3 border-t border-[var(--chef-border)] pt-3">
            {prescription.length > 0 ? (
              <>
                <button
                  type="button"
                  onClick={() => {
                    onFix(prescription)
                    setView('processed')
                  }}
                  className="flex items-center gap-1.5 rounded-md bg-[var(--chef-accent-strong)] px-3 py-1.5 text-xs font-medium text-white transition hover:brightness-110"
                >
                  <Wand2 className="size-3.5" />
                  Fix automatically
                </button>
                <span className="text-[11px] text-[var(--chef-muted)]">
                  adds {prescription.length} card{prescription.length > 1 ? 's' : ''} for{' '}
                  {problems} problem{problems > 1 ? 's' : ''}
                </span>
              </>
            ) : (
              <span className="text-[11px] text-[var(--chef-muted)]">
                {problems > 0
                  ? 'Nothing a recipe can fix here.'
                  : minor > 0
                    ? 'Only minor issues, nothing worth fixing.'
                    : 'Nothing to fix.'}
              </span>
            )}
          </div>
        )}
      </div>
    </section>
  )
}

export const SignalDoctor = memo(SignalDoctorImpl)
