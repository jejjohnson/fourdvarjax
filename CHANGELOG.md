# Changelog

## [0.1.5](https://github.com/jejjohnson/fourdvarjax/compare/v0.1.4...v0.1.5) (2026-03-01)


### Bug Fixes

* remove unused jax import in test_priors and update notebooks to NNX API ([736a97a](https://github.com/jejjohnson/fourdvarjax/commit/736a97afabe5ad8924b70d8b1ed515b34360f749))

## [0.1.4](https://github.com/jejjohnson/fourdvarjax/compare/v0.1.3...v0.1.4) (2026-03-01)


### Features

* port l96 functionality ([bb259b7](https://github.com/jejjohnson/fourdvarjax/commit/bb259b792d8e1fc6ea3c740d5f492ff5813c4537))


### Bug Fixes

* add IdentityPrior class to priors.py lost during merge with main ([6ad893f](https://github.com/jejjohnson/fourdvarjax/commit/6ad893ff811bfe5edc62b124070ad715ce5407b9))
* use type: ignore comments for N annotation in Lorenz96.__call__ to fix ty check ([d928e8f](https://github.com/jejjohnson/fourdvarjax/commit/d928e8fb033870ccb830ede9685d25d43bb30c2c))

## [0.1.3](https://github.com/jejjohnson/fourdvarjax/compare/v0.1.2...v0.1.3) (2026-03-01)


### Bug Fixes

* address review comments and CI failures (conventional commits, lint, tests) ([a65f05c](https://github.com/jejjohnson/fourdvarjax/commit/a65f05c14e93eef6cb53abc35fc268aabce9f616))
* resolve ty type-check failure in obs_interpolation_init ([aeac8e4](https://github.com/jejjohnson/fourdvarjax/commit/aeac8e46880073b42bb927d089001201cdfde034))

## [0.1.2](https://github.com/jejjohnson/fourdvarjax/compare/v0.1.1...v0.1.2) (2026-03-01)


### Features

* migrate 4dvarjax → fourdvarjax utils subpackage (L63 simulation, xarray pipeline, viz) ([93e43d7](https://github.com/jejjohnson/fourdvarjax/commit/93e43d7202220d7d295255e5eeed61badabbe63e))
* migrate 4dvarjax functionality into fourdvarjax utils subpackage ([86d6597](https://github.com/jejjohnson/fourdvarjax/commit/86d65975dab4e7bb131231d7e40b4d527ce1cc2d))


### Bug Fixes

* resolve CI failures (ruff format + xarray import at test time) ([3489196](https://github.com/jejjohnson/fourdvarjax/commit/34891960fd24b0d04ce3bbb24b8f0e2a2aa03093))
* resolve ty type-check failures and standardize.py NaN/ZeroDiv bug ([5cffd38](https://github.com/jejjohnson/fourdvarjax/commit/5cffd3834725a6b03d1b37451d0c0365b1d316e5))
* suppress ty unresolved-attribute for set_zlabel on 3D axes ([80d5442](https://github.com/jejjohnson/fourdvarjax/commit/80d544226f50307a807a60749a3b5287775d9cb5))

## [0.1.1](https://github.com/jejjohnson/fourdvarjax/compare/v0.1.0...v0.1.1) (2026-03-01)


### Features

* bootstrap fourdvarjax — full project scaffold + 4DVarNet implementation ([ab52941](https://github.com/jejjohnson/fourdvarjax/commit/ab529416d5fa436c4fbedbf0dd0a4a791ad3fe22))
* scaffold fourdvarjax repo with full 4DVarNet implementation ([c8e2aa4](https://github.com/jejjohnson/fourdvarjax/commit/c8e2aa4268378dd682ae9ef4d3bd48f744397c7e))


### Bug Fixes

* add explicit permissions to pytest.yaml workflow ([153a073](https://github.com/jejjohnson/fourdvarjax/commit/153a07352caa521cb4c2e4c91d574a26bdb81332))
* address review comments - lint, unused vars, fit() init, obs_dim removal ([4e5cdde](https://github.com/jejjohnson/fourdvarjax/commit/4e5cddee53a7ec40b1fcc36d256ceee70c7dbf50))

## Changelog
