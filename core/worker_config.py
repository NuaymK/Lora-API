INPUT_SCHEMA = {
  'dataset_url': {
      'type': str,
      'required': True
  },
  'output_directory': {
      'type': str,
      'required': True
  },
  'training_steps': {
      'type': int,
      'required': True
  },
  'model_name': {
      'type': str,
      'required': True
  },
  'model_path': {
      'type': str,
      'required': False,
      'default': "/runpod-volume/trained_models"

  },
  'instance_prompt': {
      'type': str,
      'required': True
  },
  'class_prompt': {
      'type': str,
      'required': True
  }
}
