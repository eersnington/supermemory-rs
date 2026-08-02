use crate::{
    ApiError, AppState, CreateDocumentRequest, DocumentResult, Extension, Json, OrganizationId,
    Path, State, UpsertDocument, validate_request,
};

pub(crate) async fn create_document(
    State(state): State<AppState>,
    Extension(organization): Extension<OrganizationId>,
    Json(request): Json<CreateDocumentRequest>,
) -> Result<Json<DocumentResult>, ApiError> {
    validate_request(&request)?;
    let content = storage::sanitize_content(&request.content);
    if content.is_empty() {
        return Err(ApiError::Validation(
            "content must not be empty after sanitization",
        ));
    }
    let tags = request.container_tag.clone().map_or_else(
        || {
            request
                .container_tags
                .clone()
                .unwrap_or_else(|| vec!["sm_project_default".to_owned()])
        },
        |tag| vec![tag],
    );
    let document = UpsertDocument {
        content,
        custom_id: request.custom_id,
        container_tags: tags,
        entity_context: request.entity_context,
        metadata: request.metadata,
        task_type: request.task_type.as_str().to_owned(),
        filepath: request.filepath,
        filter_by_metadata: request.filter_by_metadata,
        dreaming: request.dreaming.as_str().to_owned(),
    };
    let organization = organization.0;
    let result = state
        .writer
        .upsert_document(organization, document)
        .await
        .map_err(|_| ApiError::StorageUnavailable)?
        .map_err(ApiError::Storage)?;
    Ok(Json(DocumentResult {
        id: result.id,
        status: result.status.as_str().to_owned(),
    }))
}

pub(crate) async fn get_document(
    State(state): State<AppState>,
    Extension(organization): Extension<OrganizationId>,
    Path(id): Path<String>,
) -> Result<Json<storage::Document>, ApiError> {
    state
        .writer
        .execute(move |storage| {
            storage
                .find_document_for(&organization.0, &id)
                .map_err(ApiError::Storage)
        })
        .await
        .map_err(|_| ApiError::StorageUnavailable)??
        .map(Json)
        .ok_or(ApiError::NotFound)
}
