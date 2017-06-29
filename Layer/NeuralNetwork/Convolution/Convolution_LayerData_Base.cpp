//======================================
// 畳みこみニューラルネットワークのレイヤーデータ
//======================================
#include"stdafx.h"

#include"Convolution_LayerData_Base.h"
#include"Convolution_FUNC.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	/** コンストラクタ */
	Convolution_LayerData_Base::Convolution_LayerData_Base(const Gravisbell::GUID& guid)
		:	guid	(guid)
		,	pLayerStructure	(NULL)	/**< レイヤー構造を定義したコンフィグクラス */
		,	layerStructure	()		/**< レイヤー構造 */
		,	m_pOptimizer_neuron	(NULL)		/**< ニューロン更新用オプティマイザ */
		,	m_pOptimizer_bias	(NULL)		/**< バイアス更新用オプティマイザ */
	{
	}
	/** デストラクタ */
	Convolution_LayerData_Base::~Convolution_LayerData_Base()
	{
		if(pLayerStructure != NULL)
			delete pLayerStructure;

		if(this->m_pOptimizer_neuron)
			delete this->m_pOptimizer_neuron;
		if(this->m_pOptimizer_bias)
			delete this->m_pOptimizer_bias;
	}


	//===========================
	// 共通処理
	//===========================
	/** レイヤー固有のGUIDを取得する */
	Gravisbell::GUID Convolution_LayerData_Base::GetGUID(void)const
	{
		return this->guid;
	}

	/** レイヤー種別識別コードを取得する.
		@param o_layerCode	格納先バッファ
		@return 成功した場合0 */
	Gravisbell::GUID Convolution_LayerData_Base::GetLayerCode(void)const
	{
		Gravisbell::GUID layerCode;
		::GetLayerCode(layerCode);

		return layerCode;
	}


	//===========================
	// レイヤー設定
	//===========================
	/** 設定情報を設定 */
	ErrorCode Convolution_LayerData_Base::SetLayerConfig(const SettingData::Standard::IData& config)
	{
		Gravisbell::ErrorCode err = ERROR_CODE_NONE;

		// レイヤーコードを確認
		{
			Gravisbell::GUID config_guid;
			err = config.GetLayerCode(config_guid);
			if(err != ERROR_CODE_NONE)
				return err;

			Gravisbell::GUID layer_guid;
			err = ::GetLayerCode(layer_guid);
			if(err != ERROR_CODE_NONE)
				return err;

			if(config_guid != layer_guid)
				return ERROR_CODE_INITLAYER_DISAGREE_CONFIG;
		}

		if(this->pLayerStructure != NULL)
			delete this->pLayerStructure;
		this->pLayerStructure = config.Clone();

		// 構造体に読み込む
		this->pLayerStructure->WriteToStruct((BYTE*)&this->layerStructure);

		return ERROR_CODE_NONE;
	}

	/** レイヤーの設定情報を取得する */
	const SettingData::Standard::IData* Convolution_LayerData_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
	U32 Convolution_LayerData_Base::GetUseBufferByteCount()const
	{
		U32 bufferSize = 0;

		if(pLayerStructure == NULL)
			return 0;

		// 設定情報
		bufferSize += pLayerStructure->GetUseBufferByteCount();

		// 本体のバイト数
		bufferSize += (this->layerStructure.Output_Channel * this->layerStructure.FilterSize.x * this->layerStructure.FilterSize.y * this->layerStructure.FilterSize.z * this->layerStructure.Input_Channel) * sizeof(NEURON_TYPE);	// ニューロン係数
		bufferSize += this->layerStructure.Output_Channel * sizeof(NEURON_TYPE);	// バイアス係数

		// オプティマイザーのバイト数
		bufferSize += this->m_pOptimizer_bias->GetUseBufferByteCount();
		bufferSize += this->m_pOptimizer_neuron->GetUseBufferByteCount();

		return bufferSize;
	}


	//===========================
	// レイヤー構造
	//===========================
	/** 入力データ構造が使用可能か確認する.
		@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
		@return	使用可能な入力データ構造の場合trueが返る. */
	bool Convolution_LayerData_Base::CheckCanUseInputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		if(i_inputLayerCount == 0)
			return false;
		if(i_inputLayerCount > 1)
			return false;
		if(i_lpInputDataStruct == NULL)
			return false;
		if(i_lpInputDataStruct[0].x == 0)
			return false;
		if(i_lpInputDataStruct[0].y == 0)
			return false;
		if(i_lpInputDataStruct[0].z == 0)
			return false;
		if(i_lpInputDataStruct[0].ch == 0)
			return false;
		if(i_lpInputDataStruct[0].ch != this->layerStructure.Input_Channel)
			return false;

		return true;
	}

	/** 出力データ構造を取得する.
		@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
		@return	入力データ構造が不正な場合(x=0,y=0,z=0,ch=0)が返る. */
	IODataStruct Convolution_LayerData_Base::GetOutputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return IODataStruct(0,0,0,0);

		IODataStruct outputDataStruct;

		outputDataStruct.x  = (S32)ceilf((F32)((i_lpInputDataStruct[0].x + this->layerStructure.Padding.x*2 - (this->layerStructure.FilterSize.x - 1)) / this->layerStructure.Stride.x));
		outputDataStruct.y  = (S32)ceilf((F32)((i_lpInputDataStruct[0].y + this->layerStructure.Padding.y*2 - (this->layerStructure.FilterSize.y - 1)) / this->layerStructure.Stride.y));
		outputDataStruct.z  = (S32)ceilf((F32)((i_lpInputDataStruct[0].z + this->layerStructure.Padding.z*2 - (this->layerStructure.FilterSize.z - 1)) / this->layerStructure.Stride.z));
		outputDataStruct.ch = this->layerStructure.Output_Channel;

		return outputDataStruct;
	}

	/** 複数出力が可能かを確認する */
	bool Convolution_LayerData_Base::CheckCanHaveMultOutputLayer(void)
	{
		return false;
	}

	
	//===========================
	// 固有関数
	//===========================


	//===========================
	// オプティマイザー設定
	//===========================
	/** オプティマイザーのハイパーパラメータを変更する */
	ErrorCode Convolution_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value)
	{
		if(this->m_pOptimizer_bias)
			this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
		if(this->m_pOptimizer_neuron)
			this->m_pOptimizer_neuron->SetHyperParameter(i_parameterID, i_value);

		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode Convolution_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value)
	{
		if(this->m_pOptimizer_bias)
			this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
		if(this->m_pOptimizer_neuron)
			this->m_pOptimizer_neuron->SetHyperParameter(i_parameterID, i_value);
		
		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode Convolution_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[])
	{
		if(this->m_pOptimizer_bias)
			this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
		if(this->m_pOptimizer_neuron)
			this->m_pOptimizer_neuron->SetHyperParameter(i_parameterID, i_value);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
