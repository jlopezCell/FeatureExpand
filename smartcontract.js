const { Server, Networks, Keypair, TransactionBuilder, Operation, xdr, BASE_FEE } = require('@stellar/stellar-sdk');

// Configura la red Stellar
const server = new Server('https://horizon-testnet.stellar.org');
const sourceKeypair = Keypair.fromSecret('YOUR_SOURCE_ACCOUNT_SECRET');
const sourcePublicKey = sourceKeypair.publicKey();

async function invokeContract() {
  try {
    // Carga la cuenta fuente
    const sourceAccount = await server.loadAccount(sourcePublicKey);

    // Define el contrato y los parámetros
    const contractId = 'CBCY4PEZ23HVUI35ADFVCHZKLEQMUMB5BKYNKWPPFGQYDWH7NMC5HN67';
    const functionName = 'increment';
    const user = 'GB6IPRATZ3LZV7VI3OQZSDGRHJPBADD5PAU5BBXPWH7OZ52PCS6UOJSO';
    const email = 'jlopez@gmail.com';
    const amount = '21';
//https://developers.stellar.org/docs/learn/encyclopedia/contract-development/contract-interactions/stellar-transaction

    // Construye la transacción
    const transaction = new TransactionBuilder(sourceAccount, {
      fee: BASE_FEE,
      networkPassphrase: Networks.TESTNET,
    })
      .addOperation(Operation.invokeHostFunction({
        function: xdr.HostFunction.hostFunctionTypeInvokeContract(contractId),
        parameters: [
          new xdr.ScVal.scvString(user),
          new xdr.ScVal.scvString(email),
          new xdr.ScVal.scvU32(parseInt(amount, 10))
        ]
      }))
      .setTimeout(30)
      .build();

    // Firma la transacción
    transaction.sign(sourceKeypair);

    // Envía la transacción a la red Stellar
    const response = await server.submitTransaction(transaction);
    console.log('Respuesta de la transacción:', response);
  } catch (error) {
    console.error('Error al invocar el contrato:', error);
  }
}

invokeContract();