---
title: "#6 OAuth Token Exchange (RFC 8693) — what it is, why it exists"
slug: 6
date: 2026-05-18T18:00:00+09:00
authors: [createcentury]
tags: [auth, oauth]
---

[RFC 8693 — *OAuth 2.0 Token Exchange*](https://datatracker.ietf.org/doc/html/rfc8693) (2020) standardises "trade this token for a different token". It's the protocol that quietly underwrites most modern service-to-service authentication, identity-provider bridging, and on-behalf-of delegation.

This post is the working notebook: what the spec actually says, how the moving parts connect, and where the design decisions matter in production.

{/* truncate */}

## The problem it solves

OAuth 2.0 (RFC 6749) gives you grant types for *acquiring* tokens — authorization code, client credentials, refresh token. But it has nothing standardised for **exchanging** one token for another with different properties: different audience, narrower scope, different format (SAML → JWT), or "acting on behalf of" semantics.

Before RFC 8693, every vendor had a proprietary version:

- AWS STS `AssumeRole` / `GetFederationToken`
- Google `IAMCredentials.generateAccessToken`
- Keycloak's pre-standard token-exchange endpoint
- Auth0 / Okta custom implementations

RFC 8693 unifies these into a single token endpoint extension. Same `/token` endpoint, new `grant_type`.

## The request

```http
POST /oauth/token HTTP/1.1
Host: as.example.com
Content-Type: application/x-www-form-urlencoded

grant_type=urn:ietf:params:oauth:grant-type:token-exchange
&subject_token=<the token you have>
&subject_token_type=urn:ietf:params:oauth:token-type:access_token
&requested_token_type=urn:ietf:params:oauth:token-type:access_token
&audience=https://api.downstream.example.com
&scope=read:resources
```

Parameter glossary:

| Parameter | Purpose |
|---|---|
| `grant_type` | Always the URN `urn:ietf:params:oauth:grant-type:token-exchange` |
| `subject_token` | The token you currently hold — the **input** |
| `subject_token_type` | What `subject_token` is (access / refresh / id / SAML / JWT) |
| `requested_token_type` | What kind of token you want **back** |
| `audience` | The downstream system that'll consume the new token |
| `resource` | URI of the protected resource (alternative to audience) |
| `scope` | Requested permissions for the new token (subset of subject's) |
| `actor_token` (optional) | A second token identifying *who is doing the exchange* — see delegation below |
| `actor_token_type` (optional) | Type of `actor_token` |

## The response

```json
{
  "access_token":      "<the new token>",
  "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
  "token_type":        "Bearer",
  "expires_in":        3600,
  "scope":             "read:resources"
}
```

`issued_token_type` is mandatory and may differ from `requested_token_type` — the authorisation server tells you what it actually issued.

## Token type URNs

| URN | What it identifies |
|---|---|
| `urn:ietf:params:oauth:token-type:access_token` | Generic OAuth 2.0 access token |
| `urn:ietf:params:oauth:token-type:refresh_token` | OAuth 2.0 refresh token |
| `urn:ietf:params:oauth:token-type:id_token` | OIDC ID token |
| `urn:ietf:params:oauth:token-type:saml1` | SAML 1.1 assertion |
| `urn:ietf:params:oauth:token-type:saml2` | SAML 2.0 assertion |
| `urn:ietf:params:oauth:token-type:jwt` | JWT (RFC 7519), format-only |

These URNs decouple "what's inside the token" (access / refresh / id / assertion) from "how it's serialised" (opaque / JWT / SAML XML). Useful when bridging across identity systems.

## Delegation vs impersonation

The single most consequential choice in token exchange. Both look similar in the wire format, but the semantics — and the audit trail — are very different.

### Impersonation (no `actor_token`)

The new token *is* the subject. Downstream sees only the original principal; the fact that someone else triggered the exchange is **invisible**.

```
subject:  alice@example.com
new token claims:
  sub:    alice@example.com
  aud:    api.downstream
```

Risk: lost audit trail. Whoever exchanged on Alice's behalf is gone from the chain.

### Delegation (`actor_token` present)

The new token says "Alice's request, executed by Bob". RFC 8693 puts the actor inside the standard `act` claim ([RFC 8693 §4.1](https://datatracker.ietf.org/doc/html/rfc8693#section-4.1)):

```json
{
  "sub": "alice@example.com",
  "aud": "api.downstream",
  "act": {
    "sub": "service-foo@svc.example.com"
  }
}
```

The `act` claim can be **nested**, recording a chain of delegators if the token gets exchanged multiple times. Downstream services can authorise based on either the subject, the actor, or the full chain.

## Use cases in practice

- **Service mesh internal calls.** Front-end receives a user access token, exchanges for a short-lived service-bound token before calling backend service B. Audit trail preserved via `act`. AWS does this via STS; Google via `GenerateAccessToken`.
- **Scope reduction.** Hold a high-privilege admin token, exchange for a token narrowed to a single tenant or single API.
- **Audience binding.** Have a token with `aud=*`, exchange for `aud=specific-service` before passing it downstream — limits blast radius if the downstream service is compromised.
- **Identity-provider bridging.** Trade a SAML assertion from an enterprise IdP for an OIDC ID token usable inside a cloud-native stack.
- **Token format conversion.** Opaque ↔ JWT, often for downstream services that can validate JWTs locally without calling the auth server.

## Pitfalls

A non-exhaustive list:

1. **`aud` validation is mandatory at the consumer.** Token exchange is most powerful when the new token is *narrowly* bound to one audience. The receiving service has to actually check `aud`; otherwise narrowing was theatrical.
2. **Delegation vs impersonation is a security decision, not a default.** Picking impersonation just because it's "simpler" silently destroys the audit trail. Many auth platforms default to impersonation if `actor_token` is omitted — that default should be reviewed per-deployment.
3. **The exchange endpoint is a high-value target.** Anyone who can hit it with a valid subject token can produce derived tokens. Lock down client authentication, mTLS or signed JWT client auth recommended.
4. **Scope upgrades are not allowed by spec.** RFC 8693 §4.1: the issued token's authorisation must be *no broader* than the subject's. But many implementations don't enforce this — verify before relying on it.
5. **Chained delegation gets ugly fast.** Nested `act` claims can grow indefinitely. Set a hop limit at the AS.
6. **Replay across audiences.** If `aud` is unset or set too broadly, the same token can be reused at multiple services. Always require `aud` or `resource`.
7. **The `may_act` / actor policy is implementation-defined.** RFC 8693 doesn't tell you *who's allowed to exchange whose token*. That's left to the AS. Misconfiguration here is the source of most real-world incidents.

## Vendor mappings

How major platforms surface token exchange:

| Platform | Mechanism |
|---|---|
| AWS | STS `AssumeRole`, `AssumeRoleWithWebIdentity`, `AssumeRoleWithSAML` (predates RFC, similar semantics) |
| Google Cloud | `iam.serviceAccounts.getAccessToken` / `generateIdToken`, Workforce Identity Federation |
| Keycloak | Token exchange endpoint, gated by `realm-management:impersonation` feature flag |
| Auth0 | Token Vault / native-to-OIDC token exchange, partial RFC 8693 support |
| Okta | Token exchange OIE add-on, full RFC 8693 |

The vendor APIs diverge from the standard in detail (parameter names, supported `requested_token_type` values, `act` claim shape), but the conceptual model is the same.

## Reading list

- [RFC 8693 — OAuth 2.0 Token Exchange](https://datatracker.ietf.org/doc/html/rfc8693)
- [RFC 6749 — The OAuth 2.0 Authorization Framework](https://datatracker.ietf.org/doc/html/rfc6749) (context)
- [RFC 7519 — JSON Web Token (JWT)](https://datatracker.ietf.org/doc/html/rfc7519) (most common token format used)
- [RFC 8725 — JWT Best Current Practices](https://datatracker.ietf.org/doc/html/rfc8725) (essential for any JWT-issuing implementation)
- AWS STS [documentation](https://docs.aws.amazon.com/STS/latest/APIReference/welcome.html) — vendor-flavoured but instructive

---

*Created: 2026-05-18 / Updated: 2026-05-18*
