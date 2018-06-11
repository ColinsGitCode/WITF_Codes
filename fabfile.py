from fabric.api import local

def git_add():
    local("git add -A ")
    #local("git add * ")

def git_commit(des = "default descriptions"):
    cmd = "git commit -m \"" + des + "\""
    print(cmd)
    local(cmd)

def git_push(head = "master"):
    cmd = "git push origin " + head 
    local(cmd)

def push(des = "default descriptions", head = "master"):
    local("cp ~/.vimrc VIMRC")
    git_add()
    git_commit(des)
    git_push(head)

def pull(head = "master"):
    cmd = "git pull origin " + head
    local(cmd)
    local("cp VIMRC ~/.vimrc")
#    local("rm -rf __pycache__")


    
