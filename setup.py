from setuptools import setup, find_packages


PHASE2_PIPELINE_DATA = [
    (
        "pipeline_defs",
        ["pipeline_defs/tiktok-short-video-replication.yaml"],
    ),
    (
        "skills/meta",
        ["skills/meta/replication-scene-replacement-review.md"],
    ),
    (
        "skills/pipelines/tiktok-short-video-replication",
        [
            "skills/pipelines/tiktok-short-video-replication/delivery-director.md",
            "skills/pipelines/tiktok-short-video-replication/executive-producer.md",
            "skills/pipelines/tiktok-short-video-replication/scene-replacement-director.md",
            "skills/pipelines/tiktok-short-video-replication/source-lock-director.md",
        ],
    ),
]


setup(
    name="openmontage",
    version="0.2.0",
    description="AI-Orchestrated Video Production Platform",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "pyyaml>=6.0",
        "pydantic>=2.0",
        "jsonschema>=4.20",
        "python-dotenv>=1.0",
        "Pillow>=10.0",
        "requests>=2.31",
        "google-genai>=1.0.0",
        "openai>=2.44.0",
    ],
    extras_require={
        "replication": [
            "av==17.1.0",
            "scenedetect-headless==0.7.1",
        ],
    },
    package_data={
        "lib.replication_preprocess": ["profiles/*.yaml"],
        "schemas.replication": ["*.json"],
        "schemas.artifacts": [
            "replication_source_snapshot.schema.json",
            "scene_replacement_package.schema.json",
            "scene_replacement_delivery.schema.json",
        ],
        "schemas.tools": [
            "replication_preprocess.schema.json",
            "replication_scene_replacement_v2.schema.json",
        ],
    },
    data_files=PHASE2_PIPELINE_DATA,
)
