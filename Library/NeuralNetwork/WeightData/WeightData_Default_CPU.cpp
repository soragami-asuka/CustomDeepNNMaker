//=====================================
// 重みデータクラス.CPU制御
// デフォルト.
//=====================================
#include"stdafx.h"

#include<vector>

#include"WeightData_Default.h"

#include<Layer/NeuralNetwork/IOptimizer.h>
#include<Library/NeuralNetwork/Optimizer.h>
#include<Library/NeuralNetwork/Initializer.h>


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class WeightData_Default_CPU : public IWeightData
	{
	private:
		std::vector<F32> lpWeight;
		std::vector<F32> lpBias;

		IOptimizer* m_pOptimizer_weight;	/**< 重み更新用オプティマイザ */
		IOptimizer* m_pOptimizer_bias;		/**< バイアス更新用オプティマイザ */


	public:
		//===========================
		// コンストラクタ/デストラクタ
		//===========================
		/** コンストラクタ */
		WeightData_Default_CPU(U32 i_neuronCount, U32 i_inputCount)
			:	lpWeight			(i_neuronCount * i_inputCount)
			,	lpBias				(i_neuronCount)
			,	m_pOptimizer_weight	(NULL)
			,	m_pOptimizer_bias	(NULL)
		{
		}
		/** デストラクタ */
		virtual ~WeightData_Default_CPU()
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

			// ニューロン
			for(unsigned int weightNum=0; weightNum<this->lpWeight.size(); weightNum++)
			{
				lpWeight[weightNum] = initializer.GetParameter(i_inputCount, i_outputCount);
			}
			// バイアス
			for(unsigned int biasNum=0; biasNum<this->lpBias.size(); biasNum++)
			{
				this->lpBias[biasNum] = initializer.GetParameter(i_inputCount, i_outputCount);
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
		S64 InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize)
		{
			S64 readBufferByte = 0;
			
			// ニューロン係数
			memcpy(&this->lpWeight[0], &i_lpBuffer[readBufferByte], this->lpWeight.size() * sizeof(F32));
			readBufferByte += (int)this->lpWeight.size() * sizeof(F32);

			// バイアス
			memcpy(&this->lpBias[0], &i_lpBuffer[readBufferByte], this->lpBias.size() * sizeof(F32));
			readBufferByte += (int)this->lpBias.size() * sizeof(F32);


			// オプティマイザ
			S64 useBufferSize = 0;
			// weight
			if(this->m_pOptimizer_weight)
				delete this->m_pOptimizer_weight;
			this->m_pOptimizer_weight = CreateOptimizerFromBuffer_CPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
			readBufferByte += useBufferSize;
			// bias
			if(this->m_pOptimizer_bias)
				delete this->m_pOptimizer_bias;
			this->m_pOptimizer_bias = CreateOptimizerFromBuffer_CPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
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
			return &this->lpWeight[0];
		}
		/** Biasを取得する */
		const F32* GetBias()const
		{
			return &this->lpBias[0];
		}


		//===========================
		// 値を更新
		//===========================
		/** Weigth,Biasを設定する.
			@param	lpWeight	設定するWeightの値.
			@param	lpBias		設定するBiasの値. */
		ErrorCode SetData(const F32* i_lpWeight, const F32* i_lpBias)
		{
			memcpy(&this->lpWeight[0], i_lpWeight, sizeof(F32)*this->lpWeight.size());
			memcpy(&this->lpBias[0],   i_lpBias,   sizeof(F32)*this->lpBias.size());

			return ErrorCode::ERROR_CODE_NONE;
		}
		/** Weight,Biasを更新する.
			@param	lpDWeight	Weightの変化量.
			@param	lpDBias		Biasのh変化量. */
		ErrorCode UpdateData(const F32* i_lpDWeight, const F32* i_lpDBias)
		{
			// 誤差を反映
			if(this->m_pOptimizer_weight)
				this->m_pOptimizer_weight->UpdateParameter(&this->lpWeight[0], i_lpDWeight);
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->UpdateParameter(&this->lpBias[0],   i_lpDBias);

			return ErrorCode::ERROR_CODE_NONE;
		}


		//===========================
		// オプティマイザー設定
		//===========================
		/** オプティマイザーを変更する */
		ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[])
		{
			ChangeOptimizer_CPU(&this->m_pOptimizer_bias,   i_optimizerID, (U32)this->lpBias.size());
			ChangeOptimizer_CPU(&this->m_pOptimizer_weight, i_optimizerID, (U32)this->lpWeight.size());

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
			memcpy(&o_lpBuffer[writeBufferByte], &this->lpWeight[0], this->lpWeight.size() * sizeof(F32));
			writeBufferByte += (int)this->lpWeight.size() * sizeof(F32);
			// バイアス
			memcpy(&o_lpBuffer[writeBufferByte], &this->lpBias[0], this->lpBias.size() * sizeof(F32));
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
	IWeightData* CreateWeightData_Default_CPU(U32 i_neuronCount, U32 i_inputCount)
	{
		return new WeightData_Default_CPU(i_neuronCount, i_inputCount);
	}
}
}
}