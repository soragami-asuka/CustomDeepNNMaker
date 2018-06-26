//=====================================
// 重みデータクラス.CPU制御
// デフォルト.
//=====================================
#include"stdafx.h"

#include<vector>

#include"WeightData_WeightNormalization.h"

#include<Layer/NeuralNetwork/IOptimizer.h>
#include<Library/NeuralNetwork/Optimizer.h>
#include<Library/NeuralNetwork/Initializer.h>


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class WeightData_WeightNormalization_CPU : public IWeightData
	{
	private:
		std::vector<F32> lpWeight;
		std::vector<F32> lpBias;

		std::vector<F32> lpScale;			/**< neuron */
		std::vector<F32> lpVector;			/**< neuron*input */
		std::vector<F32> lpVectorScale;		/**< vectorの大きさ neuron */

		// 誤差用
		std::vector<F32> lpDScale;
		std::vector<F32> lpDVector;

		IOptimizer* m_pOptimizer_scale;		/**< スカラーの更新用オプティマイザ */
		IOptimizer* m_pOptimizer_vector;	/**< ベクターの更新用オプティマイザ */
		IOptimizer* m_pOptimizer_bias;		/**< バイアス更新用オプティマイザ */

		U32 neuronCount;
		U32 inputCount;

	public:
		//===========================
		// コンストラクタ/デストラクタ
		//===========================
		/** コンストラクタ */
		WeightData_WeightNormalization_CPU(U32 i_neuronCount, U32 i_inputCount)
			:	lpWeight			(i_neuronCount * i_inputCount)
			,	lpBias				(i_neuronCount)
			
			,	lpScale				(i_neuronCount)
			,	lpVector			(i_neuronCount * i_inputCount)
			,	lpVectorScale		(i_neuronCount)

			,	lpDScale			(i_neuronCount)
			,	lpDVector			(i_neuronCount * i_inputCount)

			,	m_pOptimizer_scale	(NULL)
			,	m_pOptimizer_vector	(NULL)
			,	m_pOptimizer_bias	(NULL)

			,	neuronCount			(i_neuronCount)
			,	inputCount			(i_inputCount)
		{
		}
		/** デストラクタ */
		virtual ~WeightData_WeightNormalization_CPU()
		{
			if(this->m_pOptimizer_scale)
				delete this->m_pOptimizer_scale;
			if(this->m_pOptimizer_vector)
				delete this->m_pOptimizer_vector;
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

			// 重み
			std::vector<F32> lpTmpWeight(this->lpWeight.size());
			for(unsigned int weightNum=0; weightNum<lpTmpWeight.size(); weightNum++)
			{
				lpTmpWeight[weightNum] = initializer.GetParameter(i_inputCount, i_outputCount);
			}
			// バイアス
			std::vector<F32> lpTmpBias(this->lpBias.size());
			for(unsigned int biasNum=0; biasNum<lpTmpBias.size(); biasNum++)
			{
//				lpTmpBias[biasNum] = initializer.GetParameter(i_inputCount, i_outputCount);
				lpTmpBias[biasNum] = 0.0f;
			}

			return this->SetData(&lpTmpWeight[0], &lpTmpBias[0]);
		}
		S64 InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize)
		{
			S64 readBufferByte = 0;
			
			// スケール
			memcpy(&this->lpScale[0], &i_lpBuffer[readBufferByte], this->lpScale.size() * sizeof(F32));
			readBufferByte += (int)this->lpScale.size() * sizeof(F32);
			
			// ベクター
			memcpy(&this->lpVector[0], &i_lpBuffer[readBufferByte], this->lpVector.size() * sizeof(F32));
			readBufferByte += (int)this->lpVector.size() * sizeof(F32);

			// バイアス
			memcpy(&this->lpBias[0], &i_lpBuffer[readBufferByte], this->lpBias.size() * sizeof(F32));
			readBufferByte += (int)this->lpBias.size() * sizeof(F32);


			// オプティマイザ
			S64 useBufferSize = 0;
			// scale
			if(this->m_pOptimizer_scale)
				delete this->m_pOptimizer_scale;
			this->m_pOptimizer_scale = CreateOptimizerFromBuffer_CPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
			readBufferByte += useBufferSize;
			// vector
			if(this->m_pOptimizer_vector)
				delete this->m_pOptimizer_vector;
			this->m_pOptimizer_vector = CreateOptimizerFromBuffer_CPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
			readBufferByte += useBufferSize;
			// bias
			if(this->m_pOptimizer_bias)
				delete this->m_pOptimizer_bias;
			this->m_pOptimizer_bias = CreateOptimizerFromBuffer_CPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
			readBufferByte += useBufferSize;

			// ベクターのスケールを再計算
			for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				// スケール算出
				F32 sumValue = 0.0f;
				for(U32 inputNum=0; inputNum<this->inputCount; inputNum++)
				{
					F32 value = this->lpVector[neuronNum*this->inputCount + inputNum];

					sumValue += (value * value);
				}
				this->lpVectorScale[neuronNum] = sqrtf(sumValue);
			}

			// 重みを更新
			this->UpdateWeight();

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
			// Biasをコピー
			memcpy(&this->lpBias[0],   i_lpBias,   sizeof(F32)*this->lpBias.size());

			// スケールを算出して、ベクターのサイズを1にする
			for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				// スケール算出
				F32 sumValue = 0.0f;
				for(U32 inputNum=0; inputNum<this->inputCount; inputNum++)
				{
					F32 value = i_lpWeight[neuronNum*this->inputCount + inputNum];

					sumValue += (value * value);
				}
				F32 scale = sqrtf(sumValue);
				this->lpScale[neuronNum] = scale;

				// ベクターサイズを1にする
				for(U32 inputNum=0; inputNum<this->inputCount; inputNum++)
				{
					this->lpVector[neuronNum*this->inputCount + inputNum] = i_lpWeight[neuronNum*this->inputCount + inputNum] / scale;
				}
				this->lpVectorScale[neuronNum] = 1.0f;
			}

			// 重みを再計算
			this->UpdateWeight();

			return ErrorCode::ERROR_CODE_NONE;
		}
		/** Weight,Biasを更新する.
			@param	lpDWeight	Weightの変化量.
			@param	lpDBias		Biasのh変化量. */
		ErrorCode UpdateData(const F32* i_lpDWeight, const F32* i_lpDBias)
		{
			// 誤差を計算
			for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				F32 vectorScale = this->lpVectorScale[neuronNum];

				// スケール誤差
				F32 sumValue = 0.0f;
				for(U32 inputNum=0; inputNum<this->inputCount; inputNum++)
				{
					U32 offset = neuronNum*this->inputCount + inputNum;

					sumValue += this->lpVector[offset] * i_lpDWeight[offset];
				}
				this->lpDScale[neuronNum] = sumValue / vectorScale;

				// ベクトル誤差
				for(U32 inputNum=0; inputNum<this->inputCount; inputNum++)
				{
					U32 offset = neuronNum*this->inputCount + inputNum;

					this->lpDVector[offset] = (this->lpScale[neuronNum] / vectorScale) * (i_lpDWeight[offset] - this->lpDScale[neuronNum]*this->lpVector[offset]/vectorScale);
				}
			}


			// 誤差を反映
			if(this->m_pOptimizer_scale)
				this->m_pOptimizer_scale->UpdateParameter(&this->lpScale[0], &this->lpDScale[0]);
			if(this->m_pOptimizer_vector)
				this->m_pOptimizer_vector->UpdateParameter(&this->lpVector[0], &this->lpDVector[0]);
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->UpdateParameter(&this->lpBias[0],   i_lpDBias);

			// スケールを再計算
			for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				// スケール算出
				F32 sumValue = 0.0f;
				for(U32 inputNum=0; inputNum<this->inputCount; inputNum++)
				{
					F32 value = this->lpVector[neuronNum*this->inputCount + inputNum];

					sumValue += (value * value);
				}
				this->lpVectorScale[neuronNum] = sqrtf(sumValue);
			}

			// 重みを再計算
			this->UpdateWeight();

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** Weightを更新 */
		void UpdateWeight()
		{
			for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				for(U32 inputNum=0; inputNum<this->inputCount; inputNum++)
				{
					this->lpWeight[neuronNum*this->inputCount + inputNum] = this->lpScale[neuronNum] * this->lpVector[neuronNum*this->inputCount + inputNum] / this->lpVectorScale[neuronNum];
				}
			}
		}

		//===========================
		// オプティマイザー設定
		//===========================
		/** オプティマイザーを変更する */
		ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[])
		{
			ChangeOptimizer_CPU(&this->m_pOptimizer_scale,  i_optimizerID, (U32)this->lpScale.size());
			ChangeOptimizer_CPU(&this->m_pOptimizer_vector, i_optimizerID, (U32)this->lpVector.size());
			ChangeOptimizer_CPU(&this->m_pOptimizer_bias,   i_optimizerID, (U32)this->lpBias.size());

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** オプティマイザーのハイパーパラメータを変更する */
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value)
		{
			if(this->m_pOptimizer_scale)
				this->m_pOptimizer_scale->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_vector)
				this->m_pOptimizer_vector->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);

			return ErrorCode::ERROR_CODE_NONE;
		}
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value)
		{
			if(this->m_pOptimizer_scale)
				this->m_pOptimizer_scale->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_vector)
				this->m_pOptimizer_vector->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
		
			return ErrorCode::ERROR_CODE_NONE;
		}
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[])
		{
			if(this->m_pOptimizer_scale)
				this->m_pOptimizer_scale->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_vector)
				this->m_pOptimizer_vector->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);

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
			bufferSize += sizeof(F32) * this->lpScale.size();	// スケール
			bufferSize += sizeof(F32) * this->lpVector.size();	// ベクター
			bufferSize += sizeof(F32) * this->lpBias.size();	// バイアス係数

			// オプティマイザーのバイト数
			bufferSize += this->m_pOptimizer_scale->GetUseBufferByteCount();
			bufferSize += this->m_pOptimizer_vector->GetUseBufferByteCount();
			bufferSize += this->m_pOptimizer_bias->GetUseBufferByteCount();

			return bufferSize;
		}
		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		S64 WriteToBuffer(BYTE* o_lpBuffer)const
		{
			S64 writeBufferByte = 0;

			// スケール
			memcpy(&o_lpBuffer[writeBufferByte], &this->lpScale[0], this->lpScale.size() * sizeof(F32));
			writeBufferByte += (int)this->lpScale.size() * sizeof(F32);
			// ベクター
			memcpy(&o_lpBuffer[writeBufferByte], &this->lpVector[0], this->lpVector.size() * sizeof(F32));
			writeBufferByte += (int)this->lpVector.size() * sizeof(F32);
			// バイアス
			memcpy(&o_lpBuffer[writeBufferByte], &this->lpBias[0], this->lpBias.size() * sizeof(F32));
			writeBufferByte += (int)this->lpBias.size() * sizeof(F32);

			// オプティマイザ
			// scale
			writeBufferByte += this->m_pOptimizer_scale->WriteToBuffer(&o_lpBuffer[writeBufferByte]);
			// vector
			writeBufferByte += this->m_pOptimizer_vector->WriteToBuffer(&o_lpBuffer[writeBufferByte]);
			// bias
			writeBufferByte += this->m_pOptimizer_bias->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

			return writeBufferByte;
		}
	};

	/** 重みクラスを作成する.
		デフォルト.CPU制御. */
	IWeightData* CreateWeightData_WeightNormalization_CPU(U32 i_neuronCount, U32 i_inputCount)
	{
		return new WeightData_WeightNormalization_CPU(i_neuronCount, i_inputCount);
	}
}
}
}