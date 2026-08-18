<script>
  import { fade, fly } from 'svelte/transition'
  import { cubicOut } from 'svelte/easing'
  import { login, register } from '../utils/api.js'
  import { theme } from '../stores/appStore.js'

  let mode = 'login'
  let username = ''
  let email = ''
  let password = ''
  let loading = false
  let error = ''

  async function handleSubmit() {
    error = ''
    loading = true
    try {
      if (mode === 'login') {
        await login({ username: username.trim(), password })
      } else {
        await register({
          username: username.trim(),
          email: email.trim(),
          password,
        })
      }
    } catch (err) {
      error = err.message || 'Something went wrong'
    } finally {
      loading = false
    }
  }

  function switchMode(next) {
    mode = next
    error = ''
    password = ''
    if (next === 'login') email = ''
  }

  $: initials = username.trim().slice(0, 2).toUpperCase() || 'AS'
</script>

<div class="auth-root" data-theme={$theme}>
  <div class="auth-bg">
    <div class="grid-lines"></div>
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
  </div>

  <div class="auth-shell" in:fly={{ y: 24, duration: 400, easing: cubicOut }}>
    <div class="auth-brand">
      <div class="brand-icon">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="1.5" opacity="0.8"/>
          <circle cx="12" cy="12" r="4" stroke="currentColor" stroke-width="1.5" opacity="0.6"/>
          <circle cx="12" cy="12" r="1" fill="currentColor"/>
        </svg>
      </div>
      <div>
        <div class="brand-name">ASTRAEUS</div>
        <div class="brand-sub">Space Debris Sentinel</div>
      </div>
    </div>

    <div class="auth-card">
      <div class="auth-tabs">
        <button
          class="tab"
          class:active={mode === 'login'}
          on:click={() => switchMode('login')}
        >
          Sign In
        </button>
        <button
          class="tab"
          class:active={mode === 'register'}
          on:click={() => switchMode('register')}
        >
          Register
        </button>
      </div>

      <div class="auth-eyebrow">
        {mode === 'login' ? 'Operator Access' : 'Create Account'}
      </div>
      <h1 class="auth-title">
        {mode === 'login' ? 'Mission Control Login' : 'Join the Mission'}
      </h1>
      <p class="auth-desc">
        {mode === 'login'
          ? 'Authenticate to access orbital tracking, risk analysis, and telemetry.'
          : 'First account becomes admin. New accounts start as viewers until promoted.'}
      </p>

      <form class="auth-form" on:submit|preventDefault={handleSubmit}>
        <label class="field">
          <span>Username</span>
          <input
            type="text"
            bind:value={username}
            placeholder="operator_id"
            autocomplete="username"
            required
          />
        </label>

        {#if mode === 'register'}
          <label class="field" in:fade={{ duration: 150 }}>
            <span>Email</span>
            <input
              type="email"
              bind:value={email}
              placeholder="you@agency.gov"
              autocomplete="email"
              required
            />
          </label>
        {/if}

        <label class="field">
          <span>Password</span>
          <input
            type="password"
            bind:value={password}
            placeholder="••••••••"
            autocomplete={mode === 'login' ? 'current-password' : 'new-password'}
            minlength="6"
            required
          />
        </label>

        {#if error}
          <div class="error" in:fade>{error}</div>
        {/if}

        <button class="submit" type="submit" disabled={loading}>
          {loading ? 'AUTHENTICATING...' : mode === 'login' ? 'ENTER MISSION CONTROL' : 'CREATE ACCOUNT'}
        </button>
      </form>
    </div>

    <div class="auth-footer">
      <span class="preview-badge">{initials}</span>
      <span>Secure JWT session · v2.4.1</span>
    </div>
  </div>
</div>

<style>
  .auth-root {
    position: relative;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg);
    overflow: hidden;
    padding: 24px;
  }

  .auth-bg {
    position: absolute;
    inset: 0;
    pointer-events: none;
  }

  .grid-lines {
    position: absolute;
    inset: 0;
    background-image:
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 48px 48px;
    opacity: 0.25;
  }

  .orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.15;
  }

  .orb-1 {
    width: 320px;
    height: 320px;
    top: 10%;
    left: 15%;
    background: var(--text-3);
  }

  .orb-2 {
    width: 280px;
    height: 280px;
    bottom: 10%;
    right: 12%;
    background: var(--text-4);
  }

  .auth-shell {
    position: relative;
    width: 100%;
    max-width: 420px;
    z-index: 1;
  }

  .auth-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
  }

  .brand-icon {
    width: 44px;
    height: 44px;
    border: 1px solid var(--border2);
    color: var(--text-3);
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .brand-name {
    font-family: 'Inter', sans-serif;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: var(--text);
  }

  .brand-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: var(--text-4);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
  }

  .auth-card {
    background: var(--card);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-lg);
    padding: 28px;
  }

  .auth-tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 24px;
  }

  .tab {
    flex: 1;
    height: 34px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-4);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.15s;
  }

  .tab.active {
    background: var(--surface);
    color: var(--text);
    border-color: var(--border2);
  }

  .auth-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: var(--text-4);
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }

  .auth-title {
    font-family: 'Inter', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
    margin-bottom: 10px;
  }

  .auth-desc {
    font-size: 12px;
    line-height: 1.6;
    color: var(--text-3);
    margin-bottom: 24px;
  }

  .auth-form {
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .field span {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: var(--text-4);
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  .field input {
    height: 40px;
    padding: 0 12px;
    background: var(--bg2);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    outline: none;
    transition: border-color 0.15s;
  }

  .field input:focus {
    border-color: var(--border2);
  }

  .field input::placeholder {
    color: var(--text-4);
  }

  .error {
    padding: 10px 12px;
    border: 1px solid var(--border2);
    background: var(--danger-bg);
    color: var(--text-2);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
  }

  .submit {
    height: 42px;
    margin-top: 4px;
    border: 1px solid var(--text);
    background: var(--text);
    color: var(--bg);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.14em;
    cursor: pointer;
    transition: opacity 0.15s;
  }

  .submit:hover:not(:disabled) {
    opacity: 0.9;
  }

  .submit:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .auth-footer {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    margin-top: 18px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: var(--text-4);
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }

  .preview-badge {
    width: 22px;
    height: 22px;
    border: 1px solid var(--border);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 8px;
    color: var(--text-3);
  }
</style>
