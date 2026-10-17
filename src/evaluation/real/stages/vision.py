"""Real vision assessment stages."""


class VisionSteps:
    """Assess real images only when the scenario has matching image assets."""

    def assess_vision(self, run):
        """Assess real images only when the scenario has matching image assets."""
        observed = run.observed
        if not self.config.include_vision:
            self._skip(observed, "vision", "Disabled by configuration")
            return
        if run.scenario.vision_required and not run.scenario.external_assets:
            self._skip(
                observed,
                "vision",
                "Golden vision uses deterministic detectors; no matching real image ground truth. "
                "Real external CarDD diagnostic is reported separately",
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
        run.vision = self._record(
            observed,
            "vision",
            lambda: service.assess_claim(run.case, quoted_damage_types=quoted),
            inputs={"declared_damage": [item.value for item in run.case.claim.declared_damage]},
            sources=[str(asset.path) for asset in assets],
        )
        self._capture_vision(observed, run.vision)

    @staticmethod
    def valid_image_assets(run):
        """Return the readable image assets and one reason per unusable asset."""
        from claims.assets import resolve_image_assets
        from vision.images import load_validated_image

        reasons = []
        assets = resolve_image_assets(run.case)
        if not assets:
            reasons.append("Claim has no real image assets")
        for asset in assets:
            try:
                image = load_validated_image(asset.path)
                image.close()
            except Exception as error:
                reasons.append(f"{asset.filename}: {type(error).__name__}: {error}")
        return assets, reasons
