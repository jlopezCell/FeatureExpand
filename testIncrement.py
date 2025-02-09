from typing import Optional, Union
import random
import string
from stellar_sdk import scval, MuxedAccount, Keypair, Network
from stellar_sdk.contract import AssembledTransaction, ContractClient
import sys
import json
class IncrementContractClient(ContractClient):
    def increment(
            self,
            source: Union[str, MuxedAccount],
            signer: Optional[Keypair] = None,
            email: str = "",
            points: int = 100,
            base_fee: int = 100,
            transaction_timeout: int = 300,
            submit_timeout: int = 30,
            simulate: bool = True,
            restore: bool = True,
    ) -> AssembledTransaction[int]:
        """Increment increments an internal counter, and returns the value."""
        global public_keypair
        return self.invoke(
            "increment",
            [
            scval.to_address(source),
            scval.to_string(email),
            scval.to_uint64(points)
            ],
            parse_result_xdr_fn=lambda v: scval.from_uint32(v),
            source=source,
            signer=signer,
            base_fee=base_fee,
            transaction_timeout=transaction_timeout,
            submit_timeout=submit_timeout,
            simulate=simulate,
            restore=restore,
        )


# The source account will be used to sign and send the transaction.
# GCWY3M4VRW4NXJRI7IVAU3CC7XOPN6PRBG6I5M7TAOQNKZXLT3KAH362
source_keypair = Keypair.from_secret('SCXC3D2LFPQAAKRMUFIBGIVYLEP77KTDIRTHRIINMHAEDREXOX6Y3N3Z')
##genera la clave publica de source_keypair
public_keypair = source_keypair.public_key
def parse_result_xdr_fn(v):
    if v.type == scval.SCV_U32:
        return scval.from_uint32(v)
    elif v.type == scval.SCV_U64:
        return scval.from_uint64(v)
    else:
        raise ValueError(f"Unsupported SCVal type: {v.type}")


rpc_server_url = "https://soroban-testnet.stellar.org:443"
network_passphrase = Network.TESTNET_NETWORK_PASSPHRASE
contract_address = 'CBCY4PEZ23HVUI35ADFVCHZKLEQMUMB5BKYNKWPPFGQYDWH7NMC5HN67'

client = IncrementContractClient(contract_address, rpc_server_url, network_passphrase)

if len(sys.argv) != 3:
    print("Usage: python testIncrement.py <email> <points> <output>")
    sys.exit(1)
output=None
email = sys.argv[1]
points = int(sys.argv[2])
if len(sys.argv) > 3:
    output = sys.argv[3]

transaction = client.increment(source_keypair.public_key, source_keypair, email=email, points=points)
transaction.parse_result_xdr_fn = lambda v: scval.from_uint64(v)  # Set the parsing function
result = transaction.sign_and_submit(source_keypair, force=True)

if output == None:
    ## generas un nombre aleatorio
    output = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

#salvar los resultados en un archivo con formato json

if not output.endswith('.json'):
    output += '.json'

with open(output, 'w') as f:
    json_result = json.dumps(result, indent=4)
    f.write(json_result)
print(result,output)