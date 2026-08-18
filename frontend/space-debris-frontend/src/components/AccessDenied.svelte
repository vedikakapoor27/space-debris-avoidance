<script>
  export let title = 'Access Restricted'
  export let message = 'Your role does not have permission to use this feature.'
  export let role = 'viewer'
</script>

<div class="denied">
  <div class="denied-icon">⛔</div>
  <h3>{title}</h3>
  <p>{message}</p>
  <div class="role-badge">Current role: <strong>{role}</strong></div>
</div>

<style>
  .denied {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 320px;
    padding: 40px 24px;
    text-align: center;
    border: 1px dashed var(--border);
    background: var(--card);
  }

  .denied-icon {
    font-size: 28px;
    margin-bottom: 16px;
    opacity: 0.7;
  }

  h3 {
    font-family: 'Inter', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 8px;
  }

  p {
    font-size: 12px;
    line-height: 1.6;
    color: var(--text-3);
    max-width: 320px;
    margin-bottom: 16px;
  }

  .role-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-4);
    border: 1px solid var(--border);
    padding: 6px 10px;
  }

  .role-badge strong {
    color: var(--text-2);
  }
</style>
