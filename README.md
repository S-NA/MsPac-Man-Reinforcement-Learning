To get running you will need to do the following:
```
$ pip install gymnasium 'gymnasium[atari]' 'gymnasium[accept-rom-license]'
```

Note `nix-shell` support is not complete, the optional dependencies of
`atari` and `accept-rom-license` (AutoROM) need to be added.

Theory:
At the heart of the code the Bellman equation is being used.
