# set up and dependencies
import pandas as pd
import numpy as np
import os, re
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
from flask import Flask, jsonify, render_template
from datetime import date

import spacy

nlp = spacy.load("en_core_web_sm")

