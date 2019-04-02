# LANL Earthquake Prediction

*Author:* Anders Poirel

This the Data Science Slugs team's repository for the [LANL Earthquake Prediction](https://www.kaggle.com/c/LANL-Earthquake-Prediction) competition on Kaggle.

## Getting Started

### Software Prerequisites

#### Github

Go to [GitHub](https://github.com/) and create a user account.
Message Ryan Darling or Anders Poirel on Slack so that you can be added as a collaborator on the project on GitHub. This is will be necessary for pushing your contributions to our repo.

#### Git
Get the official installer [here](https://git-scm.com/downloads). Be sure to select Git Bash at installation if you are on Windows. 
You can also use a GUI for Git but knowing command line will make things much easier for everyone down the line. If you do not know any git, [Pro Git](https://git-scm.com/book/en/v2) is a great resource to get you started. 

#### Visual Studio Code
You can use any editor you like, but to participate in live code collaboration you *will* need VS Code.
Get the official installer [here](https://code.visualstudio.com/).
Launch VS Code and in **Extensions** (*Ctrl+Shift+X*) search 'VS Live Share' and install. I also highly recommend installing the 'Python' package from Microsoft and  the 'Rainbow CSV' package, for improved Python and CSV support.

At each meeting, links to join a live code collaboration will be sent out through Slack.

#### Anaconda

*Note: this is only necessary if you intend to run the code on your local machine. You can get started without it. Also, a lot of the above software can be installed through Anaconda if you prefer.*

Download the official installer [here](https://www.anaconda.com/distribution/#download-section). Be sure to select the **Python 3.7 version**. 
I will list here as needed the Python libraries you will need to install through Anaconda.

### Setting up your environment

#### First time setup

*Note: I highly recommend reading the first 2 chapter of the book ProGit, freely available online, to get a working knowledge of git*

Create a folder to store our work on the project.

```bash
$ mkdir DataScienceSlugs
$ cd DataScienceSlugs
```

Copy the repository to your local machine, adding it as a remote - Use the name and email you used on GitHub.

```bash
$ git config --global user.name "Jonh Doe"
$ git config --global user.email  jonhdoe@example.com
$ git clone https://github.com/datascienceslugs/dss-titanic
```

#### Accessing the project

Navigate to the folder containing the project and then
run

```bash
$ git checkout testing
$ git pull origin testing
$ code [file you want to work on]
```

Which will open the file in VSCode

**All** work should be done on the testing branch, or even better, branches created specifically for the issue you want to work on.

## Contributing

See [CONTRIBUTING.md](https://github.com/datascienceslugs/dss-titanic/blob/master/CONTRIBUTING.md) for guidelines on modifying the code.
