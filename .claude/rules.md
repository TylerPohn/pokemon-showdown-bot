NEVER EVER run git commit or git push. Do not make new docs unless I ask you to. Never put anything that will be flagged by git defender in tracked files. Never point sensitive endpoints, account names, account ids, etc in tracked files. For example .envs, always use YOUR_XYZ format with all caps and snake case. Whenever we need to create aws resources, do so with aws cdk and source creds in .env in shell. Do not hardcode endpoints, creds, etc. in .sh scripts.
Always build docker images directly on the gpu. If code changes must be made, make them on my computer and prompt me that the gpu needs them
To get into GPU:
aws ssm start-session --target i-0d37b84d08727a481
sudo su - ec2-user