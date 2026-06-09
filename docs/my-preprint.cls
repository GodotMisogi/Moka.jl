%%
%% This is file `my-preprint.cls'
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{my-preprint}

% Passes and class options to the underlying article class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions

% Base it of the default article class
% Load LaTeX's article class with the `titlepage' option so that \maketitle creates a title page, not just a title block
\LoadClass[11pt]{article}

% Redefine the page margins
\RequirePackage[left=1in,right=1in,top=1in,bottom=1in]{geometry}

% \RequirePackage{hyperref}

% To modify the heading styles more thoroughly use the titlesec package
\RequirePackage{titlesec}

% Other packages to load by default
% \RequirePackage{amsmath}
% \RequirePackage{graphicx}

\def\@fnsymbol#1{\ensuremath{\ifcase#1\or *\or \dagger\or \ddagger\or
   \mathsection\or \mathparagraph\or \|\or **\or \dagger\dagger
   \or \ddagger\ddagger \else\@ctrerr\fi}}

% Adjust the title design
\newcommand{\maketitlepage}{%
  \vskip 0\p@
  \begin{center}%
    \let \footnote \thanks
    {\LARGE\bfseries \@title \par}%
    \vskip 1em%
    {\large
     \lineskip .75em%
      \begin{tabular}[t]{c}%
        \@author
        \@date
      \end{tabular}\par}%
      \vskip 1.5em%
  \end{center}\par
%   \vfil\null%
}
% This some before-and-after code that surrounds the title page.  It shouldn't need to be modified.
% I've pulled out the part the actually typesets the title page and placed it in the \maketitlepage command above.
\renewcommand\maketitle{\par
\begingroup
\renewcommand\thefootnote{\@fnsymbol\c@footnote}%
\def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
\long\def\@makefntext##1{\parindent 1em\noindent
        \hb@xt@1.8em{%
          \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
  \maketitlepage
\thispagestyle{plain}\@thanks
\endgroup
\setcounter{footnote}{0}%
\global\let\thanks\relax
\global\let\maketitle\relax
\global\let\@thanks\@empty
\global\let\@author\@empty
\global\let\@date\@empty
\global\let\@title\@empty
\global\let\title\relax
\global\let\author\relax
\global\let\date\relax
\global\let\and\relax
}
%% Abstract title: Set to Sans Serif font
%\renewenvironment{abstract}{%
%      \@beginparpenalty\@lowpenalty
%      \begin{center}%
%      \vskip 0\p@
%      \bfseries\sffamily \abstractname
%      \@endparpenalty\@M
%      \end{center}
%      \vskip -10\p@
%      }%

% Define `abstract' environment
\def\abstract{\topsep=0pt\partopsep=0pt\parsep=0pt\itemsep=0pt\relax
\trivlist\item[\hskip\labelsep
{\bfseries\abstractname}]\if!\abstractname!\hskip-\labelsep\fi}


% Section titles: Set to Sans Serif font
\renewcommand\section{\@startsection {section}{1}{\z@}%
    {-3.5ex \@plus -0.1ex \@minus -.2ex}%
    {0.2ex \@plus.01ex}%
    {\large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
    {-3.25ex\@plus -0.1ex \@minus -.2ex}%
    {0.2ex \@plus .01ex}%
    {\normalsize\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
    {-3.25ex\@plus -0.1ex \@minus -.2ex}%
    {0.2ex \@plus .01ex}%
    {\MakeUppercase\small\bfseries}}

% Table of contents: Set to Sans Serif font and tweak the spacing
% \setcounter{tocdepth}{1} % set to 2 to get next level
% \RequirePackage{tocloft}
% \setlength\cftparskip{0pt}
% \setlength\cftbeforesecskip{1pt}
% \setlength\cftaftertoctitleskip{2pt}
% \renewcommand{\cfttoctitlefont}{\Large\bfseries}
% \renewcommand{\cftsecfont}{\bfseries}
% \renewcommand{\cftsecleader}{\bfseries\cftdotfill{\cftdotsep}}
% \renewcommand{\cftsecpagefont}{\bfseries}

\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width.4\columnwidth
  \kern2.6\p@}


% Abstract


\endinput