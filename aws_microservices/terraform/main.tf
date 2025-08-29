# AWS Terraform configuration for Voicebot Microservices
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "key_pair_name" {
  description = "EC2 Key Pair name"
  type        = string
}

# VPC and Networking
resource "aws_vpc" "voicebot_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "voicebot-${var.environment}-vpc"
    Environment = var.environment
  }
}

resource "aws_internet_gateway" "voicebot_igw" {
  vpc_id = aws_vpc.voicebot_vpc.id

  tags = {
    Name        = "voicebot-${var.environment}-igw"
    Environment = var.environment
  }
}

resource "aws_subnet" "voicebot_subnet" {
  vpc_id                  = aws_vpc.voicebot_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name        = "voicebot-${var.environment}-subnet"
    Environment = var.environment
  }
}

resource "aws_route_table" "voicebot_rt" {
  vpc_id = aws_vpc.voicebot_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.voicebot_igw.id
  }

  tags = {
    Name        = "voicebot-${var.environment}-rt"
    Environment = var.environment
  }
}

resource "aws_route_table_association" "voicebot_rta" {
  subnet_id      = aws_subnet.voicebot_subnet.id
  route_table_id = aws_route_table.voicebot_rt.id
}

data "aws_availability_zones" "available" {
  state = "available"
}

# Security Groups
resource "aws_security_group" "stt_sg" {
  name_prefix = "voicebot-stt-${var.environment}"
  vpc_id      = aws_vpc.voicebot_vpc.id

  ingress {
    from_port   = 8001
    to_port     = 8001
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "voicebot-stt-${var.environment}-sg"
    Environment = var.environment
  }
}

resource "aws_security_group" "llm_sg" {
  name_prefix = "voicebot-llm-${var.environment}"
  vpc_id      = aws_vpc.voicebot_vpc.id

  ingress {
    from_port   = 8002
    to_port     = 8002
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "voicebot-llm-${var.environment}-sg"
    Environment = var.environment
  }
}

resource "aws_security_group" "tts_sg" {
  name_prefix = "voicebot-tts-${var.environment}"
  vpc_id      = aws_vpc.voicebot_vpc.id

  ingress {
    from_port   = 8003
    to_port     = 8003
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "voicebot-tts-${var.environment}-sg"
    Environment = var.environment
  }
}

# Launch Templates
resource "aws_launch_template" "stt_template" {
  name_prefix   = "voicebot-stt-${var.environment}"
  image_id      = data.aws_ami.ubuntu.id
  instance_type = "c5.xlarge"  # CPU-optimized for STT
  key_name      = var.key_pair_name

  vpc_security_group_ids = [aws_security_group.stt_sg.id]

  user_data = base64encode(templatefile("${path.module}/user_data_stt.sh", {
    service_port = 8001
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "voicebot-stt-${var.environment}"
      Environment = var.environment
      Service     = "stt"
    }
  }
}

resource "aws_launch_template" "llm_template" {
  name_prefix   = "voicebot-llm-${var.environment}"
  image_id      = data.aws_ami.ubuntu.id
  instance_type = "g4dn.xlarge"  # GPU-optimized for LLM
  key_name      = var.key_pair_name

  vpc_security_group_ids = [aws_security_group.llm_sg.id]

  user_data = base64encode(templatefile("${path.module}/user_data_llm.sh", {
    service_port = 8002
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "voicebot-llm-${var.environment}"
      Environment = var.environment
      Service     = "llm"
    }
  }
}

resource "aws_launch_template" "tts_template" {
  name_prefix   = "voicebot-tts-${var.environment}"
  image_id      = data.aws_ami.ubuntu.id
  instance_type = "g4dn.xlarge"  # High-GPU for TTS
  key_name      = var.key_pair_name

  vpc_security_group_ids = [aws_security_group.tts_sg.id]

  user_data = base64encode(templatefile("${path.module}/user_data_tts.sh", {
    service_port = 8003
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "voicebot-tts-${var.environment}"
      Environment = var.environment
      Service     = "tts"
    }
  }
}

# Auto Scaling Groups
resource "aws_autoscaling_group" "stt_asg" {
  name                = "voicebot-stt-${var.environment}-asg"
  vpc_zone_identifier = [aws_subnet.voicebot_subnet.id]
  target_group_arns   = [aws_lb_target_group.stt_tg.arn]
  health_check_type   = "ELB"
  health_check_grace_period = 300

  min_size         = 1
  max_size         = 3
  desired_capacity = 1

  launch_template {
    id      = aws_launch_template.stt_template.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "voicebot-stt-${var.environment}-asg"
    propagate_at_launch = true
  }

  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }
}

resource "aws_autoscaling_group" "llm_asg" {
  name                = "voicebot-llm-${var.environment}-asg"
  vpc_zone_identifier = [aws_subnet.voicebot_subnet.id]
  target_group_arns   = [aws_lb_target_group.llm_tg.arn]
  health_check_type   = "ELB"
  health_check_grace_period = 300

  min_size         = 1
  max_size         = 2
  desired_capacity = 1

  launch_template {
    id      = aws_launch_template.llm_template.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "voicebot-llm-${var.environment}-asg"
    propagate_at_launch = true
  }

  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }
}

resource "aws_autoscaling_group" "tts_asg" {
  name                = "voicebot-tts-${var.environment}-asg"
  vpc_zone_identifier = [aws_subnet.voicebot_subnet.id]
  target_group_arns   = [aws_lb_target_group.tts_tg.arn]
  health_check_type   = "ELB"
  health_check_grace_period = 300

  min_size         = 1
  max_size         = 2
  desired_capacity = 1

  launch_template {
    id      = aws_launch_template.tts_template.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "voicebot-tts-${var.environment}-asg"
    propagate_at_launch = true
  }

  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }
}

# Load Balancer
resource "aws_lb" "voicebot_lb" {
  name               = "voicebot-${var.environment}-lb"
  internal           = true
  load_balancer_type = "application"
  security_groups    = [aws_security_group.lb_sg.id]
  subnets            = [aws_subnet.voicebot_subnet.id]

  tags = {
    Name        = "voicebot-${var.environment}-lb"
    Environment = var.environment
  }
}

resource "aws_security_group" "lb_sg" {
  name_prefix = "voicebot-lb-${var.environment}"
  vpc_id      = aws_vpc.voicebot_vpc.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "voicebot-lb-${var.environment}-sg"
    Environment = var.environment
  }
}

# Target Groups
resource "aws_lb_target_group" "stt_tg" {
  name     = "voicebot-stt-${var.environment}-tg"
  port     = 8001
  protocol = "HTTP"
  vpc_id   = aws_vpc.voicebot_vpc.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name        = "voicebot-stt-${var.environment}-tg"
    Environment = var.environment
  }
}

resource "aws_lb_target_group" "llm_tg" {
  name     = "voicebot-llm-${var.environment}-tg"
  port     = 8002
  protocol = "HTTP"
  vpc_id   = aws_vpc.voicebot_vpc.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name        = "voicebot-llm-${var.environment}-tg"
    Environment = var.environment
  }
}

resource "aws_lb_target_group" "tts_tg" {
  name     = "voicebot-tts-${var.environment}-tg"
  port     = 8003
  protocol = "HTTP"
  vpc_id   = aws_vpc.voicebot_vpc.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name        = "voicebot-tts-${var.environment}-tg"
    Environment = var.environment
  }
}

# Load Balancer Listeners
resource "aws_lb_listener" "stt_listener" {
  load_balancer_arn = aws_lb.voicebot_lb.arn
  port              = "8001"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.stt_tg.arn
  }
}

resource "aws_lb_listener" "llm_listener" {
  load_balancer_arn = aws_lb.voicebot_lb.arn
  port              = "8002"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.llm_tg.arn
  }
}

resource "aws_lb_listener" "tts_listener" {
  load_balancer_arn = aws_lb.voicebot_lb.arn
  port              = "8003"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.tts_tg.arn
  }
}

# Data sources
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Outputs
output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.voicebot_lb.dns_name
}

output "stt_service_url" {
  description = "STT service URL"
  value       = "http://${aws_lb.voicebot_lb.dns_name}:8001"
}

output "llm_service_url" {
  description = "LLM service URL"
  value       = "http://${aws_lb.voicebot_lb.dns_name}:8002"
}

output "tts_service_url" {
  description = "TTS service URL"
  value       = "http://${aws_lb.voicebot_lb.dns_name}:8003"
}
