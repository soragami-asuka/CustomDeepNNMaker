//=====================================
// 重みデータクラス.GPU制御
// デフォルト.
//=====================================
#include<thrust/device_vector.h>

#include"WeightData_Default.h"

#include<Layer/NeuralNetwork/IOptimizer.h>
#include<Library/NeuralNetwork/Optimizer.h>
#include<Library/NeuralNetwork/Initializer.h>


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class WeightData_Default_GPU : public IWeightData
	{
	private:
		thrust::device_vector<F32> lpWeight;
		thrust::device_vector<F32> lpBias;

		IOptimizer* m_pOptimizer_weight;	/**< 重み更新用オプティマイザ */
		IOptimizer* m_pOptimizer_bias;		/**< バイアス更新用オプティマイザ */


	public:
		//===========================
		// コンストラクタ/デストラクタ
		//===========================
		/** コンストラクタ */
		WeightData_Default_GPU(U32 i_neuronCount, U32 i_inputCount)
			:	lpWeight			(i_neuronCount * i_inputCount)
			,	lpBias				(i_neuronCount)
			,	m_pOptimizer_weight	(NULL)
			,	m_pOptimizer_bias	(NULL)
		{
		}
		/** デストラクタ */
		virtual ~WeightData_Default_GPU()
		{
			if(this->m_pOptimizer_weight)
				delete this->m_pOptimizer_weight;
			if(this->m_pOptimizer_bias)
				delete this->m_pOptimizer_bias;
		}

	public:
		//===========================
		// 初期化
		//===========================
		ErrorCode Initialize(const wchar_t i_initializerID[], U32 i_inputCount, U32 i_outputCount)
		{
			auto& initializer = Gravisbell::Layer::NeuralNetwork::GetInitializerManager().GetInitializer(i_initializerID);

			thrust::host_vector<F32> lpTmpWeight(this->lpWeight.size());
			thrust::host_vector<F32> lpTmpBias(this->lpBias.size());

			for(U32 i=0; i<lpTmpWeight.size(); i++)
			{
				lpTmpWeight[i] = initializer.GetParameter(i_inputCount, i_outputCount);
			}
			for(U32 i=0; i<lpTmpBias.size(); i++)
			{
				lpTmpBias[i] = initializer.GetParameter(i_inputCount, i_outputCount);
			}

			this->lpWeight = lpTmpWeight;
			this->lpBias   = lpTmpBias;

			return ErrorCode::ERROR_CODE_NONE;
		}
		S64 InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize)
		{
			S64 readBufferByte = 0;
			
			// バッファからコピー
			// ニューロン
			cudaMemcpy(
				thrust::raw_pointer_cast(&this->lpWeight[0]),
				&i_lpBuffer[readBufferByte],
				sizeof(F32) * this->lpWeight.size(),
				cudaMemcpyHostToDevice);
			readBufferByte += sizeof(F32) * this->lpWeight.size();

			// バイアス
			cudaMemcpy(
				thrust::raw_pointer_cast(&this->lpBias[0]),
				&i_lpBuffer[readBufferByte],
				sizeof(F32) * this->lpBias.size(),
				cudaMemcpyHostToDevice);
			readBufferByte += sizeof(F32) * this->lpBias.size();


			// オプティマイザ
			S64 useBufferSize = 0;
			// neuron
			if(this->m_pOptimizer_weight)
				delete this->m_pOptimizer_weight;
			this->m_pOptimizer_weight = CreateOptimizerFromBuffer_GPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
			readBufferByte += useBufferSize;
			// bias
			if(this->m_pOptimizer_bias)
				delete this->m_pOptimizer_bias;
			this->m_pOptimizer_bias = CreateOptimizerFromBuffer_GPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
			readBufferByte += useBufferSize;


			return readBufferByte;
		}


		//===========================
		// サイズを取得
		//===========================
		/** Weightのサイズを取得する */
		U64 GetWeigthSize()const
		{
			return this->lpWeight.size();
		}
		/** Biasのサイズを取得する */
		U64 GetBiasSize()const
		{
			return this->lpBias.size();
		}


		//===========================
		// 値を取得
		//===========================
		/** Weightを取得する */
		const F32* GetWeight()const
		{
			return thrust::raw_pointer_cast(&this->lpWeight[0]);
		}
		/** Biasを取得する */
		const F32* GetBias()const
		{
			return thrust::raw_pointer_cast(&this->lpBias[0]);
		}


		//===========================
		// 値を更新
		//===========================
		/** Weigth,Biasを設定する.
			@param	lpWeight	設定するWeightの値.
			@param	lpBias		設定するBiasの値. */
		ErrorCode SetData(const F32* i_lpWeight, const F32* i_lpBias)
		{
			cudaMemcpy(thrust::raw_pointer_cast(&this->lpWeight[0]), i_lpWeight, sizeof(F32)*this->lpWeight.size(), cudaMemcpyDeviceToDevice);
			cudaMemcpy(thrust::raw_pointer_cast(&this->lpBias[0]),   i_lpBias,   sizeof(F32)*this->lpBias.size(), cudaMemcpyDeviceToDevice);

			return ErrorCode::ERROR_CODE_NONE;
		}
		/** Weight,Biasを更新する.
			@param	lpDWeight	Weightの変化量.
			@param	lpDBias		Biasのh変化量. */
		ErrorCode UpdateData(const F32* i_lpDWeight, const F32* i_lpDBias)
		{
			// 誤差を反映
			if(this->m_pOptimizer_weight)
				this->m_pOptimizer_weight->UpdateParameter(thrust::raw_pointer_cast(&this->lpWeight[0]), i_lpDWeight);
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->UpdateParameter(thrust::raw_pointer_cast(&this->lpBias[0]),   i_lpDBias);

			return ErrorCode::ERROR_CODE_NONE;
		}


		//===========================
		// オプティマイザー設定
		//===========================
		/** オプティマイザーを変更する */
		ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[])
		{
			ChangeOptimizer_GPU(&this->m_pOptimizer_bias,   i_optimizerID, (U32)this->lpBias.size());
			ChangeOptimizer_GPU(&this->m_pOptimizer_weight, i_optimizerID, (U32)this->lpWeight.size());

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** オプティマイザーのハイパーパラメータを変更する */
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value)
		{
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_weight)
				this->m_pOptimizer_weight->SetHyperParameter(i_parameterID, i_value);

			return ErrorCode::ERROR_CODE_NONE;
		}
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value)
		{
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_weight)
				this->m_pOptimizer_weight->SetHyperParameter(i_parameterID, i_value);
		
			return ErrorCode::ERROR_CODE_NONE;
		}
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[])
		{
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_weight)
				this->m_pOptimizer_weight->SetHyperParameter(i_parameterID, i_value);

			return ErrorCode::ERROR_CODE_NONE;
		}
		
		//===========================
		// レイヤー保存
		//===========================
		/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
		U64 GetUseBufferByteCount()const
		{
			U64 bufferSize = 0;

			// 本体のバイト数
			bufferSize += sizeof(F32) * this->lpWeight.size();	// 重み係数
			bufferSize += sizeof(F32) * this->lpBias.size();	// バイアス係数

			// オプティマイザーのバイト数
			bufferSize += this->m_pOptimizer_weight->GetUseBufferByteCount();
			bufferSize += this->m_pOptimizer_bias->GetUseBufferByteCount();

			return bufferSize;
		}
		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		S64 WriteToBuffer(BYTE* o_lpBuffer)const
		{
			S64 writeBufferByte = 0;

			// ニューロン係数
			cudaMemcpy(&o_lpBuffer[writeBufferByte], thrust::raw_pointer_cast(&this->lpWeight[0]), this->lpWeight.size() * sizeof(F32), cudaMemcpyDeviceToHost);
			writeBufferByte += (int)this->lpWeight.size() * sizeof(F32);
			// バイアス
			cudaMemcpy(&o_lpBuffer[writeBufferByte], thrust::raw_pointer_cast(&this->lpBias[0]), this->lpBias.size() * sizeof(F32), cudaMemcpyDeviceToHost);
			writeBufferByte += (int)this->lpBias.size() * sizeof(F32);

			// オプティマイザ
			// weight
			writeBufferByte += this->m_pOptimizer_weight->WriteToBuffer(&o_lpBuffer[writeBufferByte]);
			// bias
			writeBufferByte += this->m_pOptimizer_bias->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

			return writeBufferByte;
		}
	};

	/** 重みクラスを作成する.
		デフォルト.CPU制御. */
	IWeightData* CreateWeightData_Default_GPU(U32 i_neuronCount, U32 i_inputCount)
	{
		return new WeightData_Default_GPU(i_neuronCount, i_inputCount);
	}
}
}
}