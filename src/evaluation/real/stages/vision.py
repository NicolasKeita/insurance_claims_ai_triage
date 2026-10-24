"""Real vision assessment stages."""


class VisionSteps:
    """Run real detectors on configured image assets with explicit provenance."""

    def assess_vision(self, run):
        """Run real detectors on configured image assets with explicit provenance."""
        observed = run.observed
        if not self.config.include_vision:
            self._skip(observed, "vision", "Disabled by configuration")
            return
        if run.scenario.vision_required and not run.scenario.external_assets:
            self._skip(
                observed,
                "vision",
                "No annotated visual reference images configured for real detector execution. "
                "The deterministic geometry fixture cannot replace a reference image.",
            )
            return
        if not self._available(observed, "vision", ["damage_vision", "part_vision"]):
            return
        assets, reasons = self.valid_image_assets(run)
        if reasons:
            self._skip(observed, "vision", "; ".join(reasons))
            return
        from claims.consistency import map_quote_damage_types
        from vision.multimodal_service import ClaimMultimodalVisionService

        service = ClaimMultimodalVisionService(self.resources["damage_detector"], self.resources["part_detector"])
        quoted = map_quote_damage_types(run.quote)[0] if run.quote is not None else ()
        try:
            run.vision = self._record(
                observed,
                "vision",
                lambda: service.assess_claim(run.case, quoted_damage_types=quoted),
                inputs={"declared_damage": [item.value for item in run.case.claim.declared_damage]},
                sources=[str(asset.path) for asset in assets],
            )
        finally:
            runtime = self.resources.get("vision_runtime")
            if runtime is not None:
                runtime.release()
        self._capture_vision(observed, run.vision)

    @staticmethod
    def valid_image_assets(run):
        """Return the readable image assets and one reason per unusable asset."""
        from claims.assets import resolve_image_assets
        from vision.images import load_validated_image

        reasons = []
        assets = resolve_image_assets(run.case)
        if not assets:
            reasons.append("Claim has no image assets for detector inference")
        for asset in assets:
            try:
                image = load_validated_image(asset.path)
                image.close()
            except Exception as error:
                reasons.append(f"{asset.filename}: {type(error).__name__}: {error}")
        return assets, reasons

    def _capture_vision(self, observation, vision):
        if vision is None:
            return
        observation.vision = vision.model_dump(mode="json")
        observation.facts["vision"] = observation.vision
        observation.model_versions.update(
            damage_vision=vision.damage_model_metadata.version, part_vision=vision.part_model_metadata.version
        )
        observation.policy_versions["fusion"] = vision.fusion_policy.version
        observation.provenance["vision"] = {
            "images": [image.filename for image in vision.images],
            "damage_model": vision.damage_model_metadata.model_dump(mode="json"),
            "part_model": vision.part_model_metadata.model_dump(mode="json"),
            "geometry": "Production schema validates normalized geometry and source identity",
        }
