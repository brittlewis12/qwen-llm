use super::ApiError;
use crate::serve::http::HttpRequest;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BrowserOrigin {
    scheme: String,
    host: String,
    port: u16,
}

impl BrowserOrigin {
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        let invalid =
            || format!("expected an exact http:// or https:// origin without a path: {value:?}");
        let (scheme, authority) = value.split_once("://").ok_or_else(invalid)?;
        let default_port = match scheme {
            "http" => 80,
            "https" => 443,
            _ => return Err(invalid()),
        };
        let (host, port) = parse_authority(authority, default_port).ok_or_else(invalid)?;
        Ok(Self {
            scheme: scheme.into(),
            host,
            port,
        })
    }

    fn matches_host(&self, host: &str) -> bool {
        let default_port = if self.scheme == "https" { 443 } else { 80 };
        parse_authority(host, default_port).as_ref() == Some(&(self.host.clone(), self.port))
    }
}

#[derive(Default)]
pub(crate) struct BrowserAccess {
    allowed: Vec<BrowserOrigin>,
}

impl BrowserAccess {
    pub(crate) fn new(allowed: Vec<BrowserOrigin>) -> Self {
        Self { allowed }
    }

    pub(super) fn check(&self, request: &HttpRequest) -> Result<(), ApiError> {
        let host = request.host.as_deref().unwrap_or_default();
        let local = parse_authority(host, 80)
            .filter(|(name, _)| {
                name == "localhost"
                    || name
                        .parse::<std::net::IpAddr>()
                        .is_ok_and(|ip| ip.is_loopback())
            })
            .map(|(host, port)| BrowserOrigin {
                scheme: "http".into(),
                host,
                port,
            });
        let candidates = || {
            local.iter().chain(
                self.allowed
                    .iter()
                    .filter(|origin| origin.matches_host(host)),
            )
        };
        if candidates().next().is_none() {
            return Err(ApiError::new(
                403,
                "untrusted_host",
                "Lens requires a loopback Host or an explicitly allowed origin authority",
            ));
        }
        if let Some(origin) = &request.origin {
            let origin = BrowserOrigin::parse(origin).ok();
            if !candidates().any(|candidate| Some(candidate) == origin.as_ref()) {
                return Err(ApiError::new(
                    403,
                    "untrusted_origin",
                    "Cross-origin browser access to Lens history is not allowed",
                ));
            }
        }
        Ok(())
    }
}

fn parse_authority(authority: &str, default_port: u16) -> Option<(String, u16)> {
    let (host, port) = if let Some(ipv6) = authority.strip_prefix('[') {
        let (host, suffix) = ipv6.split_once(']')?;
        let ip: std::net::Ipv6Addr = host.parse().ok()?;
        (
            ip.to_string(),
            if suffix.is_empty() {
                None
            } else {
                Some(suffix.strip_prefix(':')?)
            },
        )
    } else {
        let (host, port) = authority
            .split_once(':')
            .map_or((authority, None), |(host, port)| (host, Some(port)));
        let name = host.strip_suffix('.').unwrap_or(host).to_ascii_lowercase();
        if name.len() > 253
            || !name.split('.').all(|label| {
                !label.is_empty()
                    && label.len() <= 63
                    && label
                        .as_bytes()
                        .first()
                        .is_some_and(u8::is_ascii_alphanumeric)
                    && label
                        .as_bytes()
                        .last()
                        .is_some_and(u8::is_ascii_alphanumeric)
                    && label
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-')
            })
        {
            return None;
        }
        (name, port)
    };
    let port = match port {
        None => default_port,
        Some(port) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => {
            port.parse().ok()?
        }
        _ => return None,
    };
    (port != 0).then_some((host, port))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_configuration_is_exact_and_normalized() {
        assert_eq!(
            BrowserOrigin::parse("https://MACHINE.tail.ts.net:443").unwrap(),
            BrowserOrigin::parse("https://machine.tail.ts.net").unwrap()
        );
        for value in [
            "null",
            "machine.tail.ts.net",
            "ftp://machine",
            "https://*.ts.net",
            "https://user@machine",
            "https://machine/",
            "https://machine/path",
            "https://machine?query",
            "https://machine#fragment",
            "https://machine:0",
            "https://machine:65536",
            "https://machine:",
            "https://machine other",
            "https://-machine",
            "https://machine..ts.net",
            "https://[::1]junk",
        ] {
            assert!(BrowserOrigin::parse(value).is_err(), "{value}");
        }
    }
}
