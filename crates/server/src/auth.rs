use super::{
    AppState, AuthError, ConnectInfo, ConstantTimeEq, Digest, Next, OrganizationId, Request,
    Response, Sha256, SocketAddr, State,
};

pub(crate) async fn authenticate(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    mut request: Request,
    next: Next,
) -> Result<Response, AuthError> {
    let supplied = request
        .headers()
        .get(http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    let organization = supplied.and_then(|key| {
        let supplied_hash: [u8; 32] = Sha256::digest(key).into();
        state
            .api_keys
            .iter()
            .find(|(expected, _)| bool::from(supplied_hash.ct_eq(expected)))
            .map(|(_, organization)| organization.clone())
    });
    let has_session_material = request.headers().contains_key(http::header::COOKIE);
    let organization = organization.or_else(|| {
        (supplied.is_none() && !has_session_material && peer.ip().is_loopback())
            .then(|| state.local_org_id.clone())
    });
    if let Some(organization) = organization {
        request
            .extensions_mut()
            .insert(OrganizationId(organization));
        Ok(next.run(request).await)
    } else {
        Err(AuthError)
    }
}
