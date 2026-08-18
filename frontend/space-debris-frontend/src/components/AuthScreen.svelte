<script>
  import { fade, fly } from 'svelte/transition'
  import { cubicOut } from 'svelte/easing'
  import { login, register } from '../utils/api.js'
  import { theme } from '../stores/appStore.js'
  import { roleLabel } from '../utils/permissions.js'

  let mode = 'login'
  let username = ''
  let email = ''
  let password = ''
  let confirmPassword = ''
  let showPassword = false
  let acceptTerms = false
  let loading = false
  let error = ''

  const roles = [
    {
      id: 'admin',
      title: 'Administrator',
      desc: 'Full access — manage users, clear history, run predictions.',
      badge: 'ADMIN',
    },
    {
      id: 'operator',
      title: 'Operator',
      desc: 'Run AI predictions, live telemetry, and view all mission history.',
      badge: 'OPS',
    },
    {
      id: 'viewer',
      title: 'Viewer',
      desc: 'Read-only access to dashboard, conjunctions, and your own history.',
      badge: 'VIEW',
    },
  ]

  $: passwordChecks = {
    length: password.length >= 8,
    upper: /[A-Z]/.test(password),
    lower: /[a-z]/.test(password),
    number: /[0-9]/.test(password),
    special: /[^A-Za-z0-9]/.test(password),
  }

  $: passwordScore = Object.values(passwordChecks).filter(Boolean).length
  $: passwordStrong = passwordScore >= 4
  $: passwordsMatch = password === confirmPassword && confirmPassword.length > 0
  $: usernameValid = /^[a-zA-Z0-9_]{3,20}$/.test(username.trim())
  $: emailValid = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())

  $: canRegister = usernameValid && emailValid && passwordStrong && passwordsMatch && acceptTerms
  $: canLogin = username.trim().length >= 3 && password.length >= 6

  function validateBeforeSubmit() {
    if (mode === 'login') {
      if (!canLogin) {
        error = 'Enter a valid username and password (min 6 characters).'
        return false
      }
      return true
    }

    if (!usernameValid) {
      error = 'Username must be 3–20 characters (letters, numbers, underscore only).'
      return false
    }
    if (!emailValid) {
      error = 'Enter a valid email address.'
      return false
    }
    if (!passwordStrong) {
      error = 'Password is too weak. Meet at least 4 of the security requirements.'
      return false
    }
    if (!passwordsMatch) {
      error = 'Passwords do not match.'
      return false
    }
    if (!acceptTerms) {
      error = 'You must accept the terms to create an account.'
      return false
    }
    return true
  }

  async function handleSubmit() {
    error = ''
    if (!validateBeforeSubmit()) return

    loading = true
    try {
      if (mode === 'login') {
        await login({ username: username.trim(), password })
      } else {
        await register({
          username: username.trim(),
          email: email.trim().toLowerCase(),
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
    confirmPassword = ''
    acceptTerms = false
    showPassword = false
    if (next === 'login') email = ''
  }

  function strengthLabel(score) {
    if (score <= 1) return 'Very weak'
    if (score === 2) return 'Weak'
    if (score === 3) return 'Fair'
    if (score === 4) return 'Good'
    return 'Strong'
  }
</script>

<div class="auth-page" data-theme={$theme}>
  <div class="auth-bg" aria-hidden="true">
    <div class="grid"></div>
    <div class="glow glow-a"></div>
    <div class="glow glow-b"></div>
  </div>

  <header class="auth-topbar">
    <div class="brand-mini">
      <span class="brand-dot"></span>
      <span>ASTRAEUS</span>
    </div>
    <button
      class="theme-toggle"
      type="button"
      aria-label="Toggle theme"
      on:click={() => theme.update((t) => (t === 'dark' ? 'light' : 'dark'))}
    >
      {#if $theme === 'dark'}
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
          <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
        </svg>
        Light
      {:else}
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
        </svg>
        Dark
      {/if}
    </button>
  </header>

  <main class="auth-layout" in:fly={{ y: 20, duration: 350, easing: cubicOut }}>
    <section class="auth-info">
      <div class="info-badge">Secure Mission Access</div>
      <h1>Space Debris<br />Sentinel System</h1>
      <p class="info-lead">
        Authenticated access to orbital tracking, collision risk analysis, and mission telemetry.
      </p>

      <div class="security-list">
        <div class="security-item">
          <span class="check">✓</span>
          <span>JWT-secured API sessions</span>
        </div>
        <div class="security-item">
          <span class="check">✓</span>
          <span>Role-based access control</span>
        </div>
        <div class="security-item">
          <span class="check">✓</span>
          <span>Password strength enforcement</span>
        </div>
      </div>

      <div class="roles-block">
        <h2>Access Roles</h2>
        <p class="roles-note">
          {#if mode === 'register'}
            New accounts start as <strong>Viewer</strong>. The first account becomes <strong>Administrator</strong>.
          {:else}
            Your permissions depend on your assigned role after login.
          {/if}
        </p>
        <div class="role-cards">
          {#each roles as role}
            <div class="role-card">
              <div class="role-top">
                <span class="role-badge">{role.badge}</span>
                <span class="role-name">{role.title}</span>
              </div>
              <p>{role.desc}</p>
            </div>
          {/each}
        </div>
      </div>
    </section>

    <section class="auth-card">
      <div class="card-tabs">
        <button class="tab" class:active={mode === 'login'} type="button" on:click={() => switchMode('login')}>
          Sign In
        </button>
        <button class="tab" class:active={mode === 'register'} type="button" on:click={() => switchMode('register')}>
          Create Account
        </button>
      </div>

      <h2>{mode === 'login' ? 'Welcome back' : 'Create your account'}</h2>
      <p class="card-sub">
        {mode === 'login'
          ? 'Enter your credentials to access mission control.'
          : 'Set a strong password. Roles are assigned automatically for security.'}
      </p>

      <form class="auth-form" on:submit|preventDefault={handleSubmit}>
        <label class="field">
          <span>Username</span>
          <input
            type="text"
            bind:value={username}
            placeholder="e.g. mission_ops"
            autocomplete="username"
            required
          />
          {#if mode === 'register' && username && !usernameValid}
            <span class="field-hint bad">3–20 chars, letters, numbers, underscore only</span>
          {/if}
        </label>

        {#if mode === 'register'}
          <label class="field" in:fade={{ duration: 120 }}>
            <span>Email</span>
            <input
              type="email"
              bind:value={email}
              placeholder="you@agency.gov"
              autocomplete="email"
              required
            />
            {#if email && !emailValid}
              <span class="field-hint bad">Enter a valid email address</span>
            {/if}
          </label>
        {/if}

        <label class="field">
          <span>Password</span>
          <div class="password-wrap">
            <input
              type={showPassword ? 'text' : 'password'}
              bind:value={password}
              placeholder="Enter password"
              autocomplete={mode === 'login' ? 'current-password' : 'new-password'}
              required
            />
            <button class="eye-btn" type="button" on:click={() => (showPassword = !showPassword)}>
              {showPassword ? 'Hide' : 'Show'}
            </button>
          </div>
        </label>

        {#if mode === 'register'}
          <label class="field" in:fade={{ duration: 120 }}>
            <span>Confirm Password</span>
            <input
              type={showPassword ? 'text' : 'password'}
              bind:value={confirmPassword}
              placeholder="Re-enter password"
              autocomplete="new-password"
              required
            />
            {#if confirmPassword && !passwordsMatch}
              <span class="field-hint bad">Passwords do not match</span>
            {/if}
          </label>

          <div class="strength-box" in:fade={{ duration: 120 }}>
            <div class="strength-top">
              <span>Password strength</span>
              <span class:good={passwordStrong}>{strengthLabel(passwordScore)}</span>
            </div>
            <div class="strength-bar">
              <div class="strength-fill" style="width: {(passwordScore / 5) * 100}%"></div>
            </div>
            <ul class="checks">
              <li class:ok={passwordChecks.length}>At least 8 characters</li>
              <li class:ok={passwordChecks.upper}>One uppercase letter</li>
              <li class:ok={passwordChecks.lower}>One lowercase letter</li>
              <li class:ok={passwordChecks.number}>One number</li>
              <li class:ok={passwordChecks.special}>One special character</li>
            </ul>
          </div>

          <label class="terms" in:fade={{ duration: 120 }}>
            <input type="checkbox" bind:checked={acceptTerms} />
            <span>I agree to secure usage policies and understand my role is assigned by an administrator.</span>
          </label>
        {/if}

        {#if error}
          <div class="error" in:fade>{error}</div>
        {/if}

        <button
          class="submit"
          type="submit"
          disabled={loading || (mode === 'login' ? !canLogin : !canRegister)}
        >
          {#if loading}
            {mode === 'login' ? 'Signing in...' : 'Creating account...'}
          {:else}
            {mode === 'login' ? 'Sign In' : 'Create Account'}
          {/if}
        </button>
      </form>

      <div class="card-footer">
        <span>🔒 Encrypted session</span>
        <span>·</span>
        <span>v2.4.1</span>
      </div>
    </section>
  </main>
</div>

<style>
  .auth-page {
    --auth-bg: #050505;
    --auth-surface: #111111;
    --auth-card: #0d0d0d;
    --auth-border: #2b2b2b;
    --auth-text: #ffffff;
    --auth-muted: #a8a8a8;
    --auth-faint: #666666;
    --auth-accent: #ffffff;
    --auth-accent-text: #050505;
    --auth-good: #4ade80;
    --auth-bad: #f87171;

    min-height: 100vh;
    background: var(--auth-bg);
    color: var(--auth-text);
    font-family: 'Inter', sans-serif;
    position: relative;
    overflow-x: hidden;
  }

  .auth-page[data-theme='light'] {
    --auth-bg: #f7f7f8;
    --auth-surface: #ffffff;
    --auth-card: #ffffff;
    --auth-border: #e2e2e7;
    --auth-text: #0a0a0a;
    --auth-muted: #4b4b52;
    --auth-faint: #8b8b95;
    --auth-accent: #0a0a0a;
    --auth-accent-text: #ffffff;
    --auth-good: #16a34a;
    --auth-bad: #dc2626;
  }

  .auth-bg {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
  }

  .grid {
    position: absolute;
    inset: 0;
    background-image:
      linear-gradient(var(--auth-border) 1px, transparent 1px),
      linear-gradient(90deg, var(--auth-border) 1px, transparent 1px);
    background-size: 56px 56px;
    opacity: 0.18;
  }

  .glow {
    position: absolute;
    border-radius: 50%;
    filter: blur(90px);
    opacity: 0.12;
  }

  .glow-a {
    width: 360px;
    height: 360px;
    top: 8%;
    left: 8%;
    background: var(--auth-muted);
  }

  .glow-b {
    width: 300px;
    height: 300px;
    bottom: 10%;
    right: 10%;
    background: var(--auth-faint);
  }

  .auth-topbar {
    position: relative;
    z-index: 2;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 28px;
  }

  .brand-mini {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
  }

  .brand-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--auth-accent);
  }

  .theme-toggle {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    height: 36px;
    padding: 0 14px;
    border: 1px solid var(--auth-border);
    background: var(--auth-surface);
    color: var(--auth-muted);
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .theme-toggle:hover {
    color: var(--auth-text);
    border-color: var(--auth-muted);
  }

  .auth-layout {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: 1.1fr 0.9fr;
    gap: 28px;
    max-width: 1080px;
    margin: 0 auto;
    padding: 12px 28px 40px;
    align-items: start;
  }

  .auth-info h1 {
    font-size: clamp(32px, 4vw, 44px);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.03em;
    margin: 14px 0 16px;
  }

  .info-badge {
    display: inline-flex;
    padding: 6px 10px;
    border: 1px solid var(--auth-border);
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    color: var(--auth-muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  .info-lead {
    font-size: 15px;
    line-height: 1.65;
    color: var(--auth-muted);
    max-width: 520px;
    margin-bottom: 22px;
  }

  .security-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 28px;
  }

  .security-item {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    color: var(--auth-muted);
  }

  .check {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: 1px solid var(--auth-border);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    color: var(--auth-good);
    flex-shrink: 0;
  }

  .roles-block h2 {
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin-bottom: 8px;
  }

  .roles-note {
    font-size: 13px;
    line-height: 1.6;
    color: var(--auth-muted);
    margin-bottom: 14px;
  }

  .roles-note strong {
    color: var(--auth-text);
    font-weight: 600;
  }

  .role-cards {
    display: grid;
    gap: 10px;
  }

  .role-card {
    padding: 14px 16px;
    border: 1px solid var(--auth-border);
    background: color-mix(in srgb, var(--auth-surface) 88%, transparent);
    border-radius: 12px;
  }

  .role-top {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
  }

  .role-badge {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 4px 8px;
    border-radius: 6px;
    background: var(--auth-accent);
    color: var(--auth-accent-text);
  }

  .role-name {
    font-size: 13px;
    font-weight: 600;
  }

  .role-card p {
    font-size: 12px;
    line-height: 1.55;
    color: var(--auth-muted);
    margin: 0;
  }

  .auth-card {
    background: var(--auth-card);
    border: 1px solid var(--auth-border);
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.18);
  }

  .auth-page[data-theme='light'] .auth-card {
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.06);
  }

  .card-tabs {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 22px;
    padding: 4px;
    border: 1px solid var(--auth-border);
    border-radius: 12px;
    background: color-mix(in srgb, var(--auth-surface) 70%, transparent);
  }

  .tab {
    height: 38px;
    border: none;
    border-radius: 8px;
    background: transparent;
    color: var(--auth-muted);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .tab.active {
    background: var(--auth-accent);
    color: var(--auth-accent-text);
  }

  .auth-card h2 {
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
  }

  .card-sub {
    font-size: 13px;
    line-height: 1.6;
    color: var(--auth-muted);
    margin-bottom: 22px;
  }

  .auth-form {
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: 7px;
  }

  .field span {
    font-size: 12px;
    font-weight: 600;
    color: var(--auth-text);
  }

  .field input {
    height: 44px;
    padding: 0 14px;
    border: 1px solid var(--auth-border);
    border-radius: 10px;
    background: var(--auth-surface);
    color: var(--auth-text);
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    outline: none;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
  }

  .field input:focus {
    border-color: var(--auth-muted);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--auth-muted) 18%, transparent);
  }

  .field input::placeholder {
    color: var(--auth-faint);
  }

  .password-wrap {
    position: relative;
  }

  .password-wrap input {
    width: 100%;
    padding-right: 72px;
  }

  .eye-btn {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    height: 30px;
    padding: 0 10px;
    border: 1px solid var(--auth-border);
    border-radius: 8px;
    background: transparent;
    color: var(--auth-muted);
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
  }

  .field-hint {
    font-size: 11px;
    font-weight: 500;
  }

  .field-hint.bad {
    color: var(--auth-bad);
  }

  .strength-box {
    padding: 14px;
    border: 1px solid var(--auth-border);
    border-radius: 12px;
    background: color-mix(in srgb, var(--auth-surface) 80%, transparent);
  }

  .strength-top {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 8px;
  }

  .strength-top .good {
    color: var(--auth-good);
  }

  .strength-bar {
    height: 6px;
    border-radius: 999px;
    background: var(--auth-border);
    overflow: hidden;
    margin-bottom: 10px;
  }

  .strength-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--auth-bad), #fbbf24, var(--auth-good));
    transition: width 0.2s ease;
  }

  .checks {
    list-style: none;
    display: grid;
    gap: 6px;
    margin: 0;
    padding: 0;
  }

  .checks li {
    font-size: 11px;
    color: var(--auth-faint);
    position: relative;
    padding-left: 18px;
  }

  .checks li::before {
    content: '○';
    position: absolute;
    left: 0;
    color: var(--auth-faint);
  }

  .checks li.ok {
    color: var(--auth-good);
  }

  .checks li.ok::before {
    content: '✓';
    color: var(--auth-good);
  }

  .terms {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    font-size: 12px;
    line-height: 1.5;
    color: var(--auth-muted);
    cursor: pointer;
  }

  .terms input {
    margin-top: 3px;
    accent-color: var(--auth-accent);
  }

  .error {
    padding: 12px 14px;
    border-radius: 10px;
    border: 1px solid color-mix(in srgb, var(--auth-bad) 40%, var(--auth-border));
    background: color-mix(in srgb, var(--auth-bad) 10%, transparent);
    color: var(--auth-bad);
    font-size: 13px;
    font-weight: 500;
  }

  .submit {
    height: 46px;
    margin-top: 4px;
    border: none;
    border-radius: 10px;
    background: var(--auth-accent);
    color: var(--auth-accent-text);
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
    transition: opacity 0.15s ease, transform 0.15s ease;
  }

  .submit:hover:not(:disabled) {
    opacity: 0.92;
    transform: translateY(-1px);
  }

  .submit:disabled {
    opacity: 0.45;
    cursor: not-allowed;
    transform: none;
  }

  .card-footer {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-top: 18px;
    font-size: 11px;
    color: var(--auth-faint);
    font-weight: 500;
  }

  @media (max-width: 900px) {
    .auth-layout {
      grid-template-columns: 1fr;
      padding: 0 18px 32px;
    }

    .auth-info {
      order: 2;
    }

    .auth-card {
      order: 1;
    }

    .role-cards {
      grid-template-columns: 1fr;
    }
  }
</style>
