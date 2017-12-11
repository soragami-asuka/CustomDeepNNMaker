//=====================================
// パラメータ初期化クラス.
// 基本構造. 大抵の場合はこれを派生しておくと楽に作れる
//=====================================
#include"stdafx.h"

#include"Initializer_base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;



/** コンストラクタ */
Initializer_base::Initializer_base()
{
}
/** デストラクタ1 */
Initializer_base::~Initializer_base()
{
}


//===========================
// パラメータの値を取得
//===========================

/** パラメータの値を取得する.
	@param	i_inputStruct	入力構造.
	@param	i_outputStruct	出力構造. */
F32 Initializer_base::GetParameter(const IODataStruct& i_inputStruct, const IODataStruct& i_outputStruct)
{
	return this->GetParameter(i_inputStruct.GetDataCount(), i_outputStruct.GetDataCount());
}
/** パラメータの値を取得する.
	@param	i_inputStruct	入力構造.
	@param	i_inputCH		入力チャンネル数.
	@param	i_outputStruct	出力構造.
	@param	i_outputCH		出力チャンネル数. */
F32 Initializer_base::GetParameter(const Vector3D<S32>& i_inputStruct, U32 i_inputCH, const Vector3D<S32>& i_outputStruct, U32 i_outputCH)
{
	return this->GetParameter(
		IODataStruct(i_inputCH,  i_inputStruct.x,  i_inputStruct.y,  i_inputStruct.z),
		IODataStruct(i_outputCH, i_outputStruct.x, i_outputStruct.y, i_outputStruct.z) );
}

/** パラメータの値を取得する.
	@param	i_inputCount		入力信号数.
	@param	i_outputCount		出力信号数.
	@param	i_parameterCount	設定するパラメータ数.
	@param	o_lpParameter		設定するパラメータの配列. */
ErrorCode Initializer_base::GetParameter(U32 i_inputCount, U32 i_outputCount, U32 i_parameterCount, F32 o_lpParameter[])
{
	for(U32 i=0; i<i_parameterCount; i++)
	{
		o_lpParameter[i] = this->GetParameter(i_inputCount, i_outputCount);
	}

	return ErrorCode::ERROR_CODE_NONE;
}
/** パラメータの値を取得する.
	@param	i_inputStruct		入力構造.
	@param	i_outputStruct		出力構造.
	@param	i_parameterCount	設定するパラメータ数.
	@param	o_lpParameter		設定するパラメータの配列. */
ErrorCode Initializer_base::GetParameter(const IODataStruct& i_inputStruct, const IODataStruct& i_outputStruct, U32 i_parameterCount, F32 o_lpParameter[])
{
	return this->GetParameter(i_inputStruct.GetDataCount(), i_outputStruct.GetDataCount(), i_parameterCount, o_lpParameter);
}
/** パラメータの値を取得する.
	@param	i_inputStruct		入力構造.
	@param	i_inputCH			入力チャンネル数.
	@param	i_outputStruct		出力構造.
	@param	i_outputCH			出力チャンネル数.
	@param	i_parameterCount	設定するパラメータ数.
	@param	o_lpParameter		設定するパラメータの配列. */
ErrorCode Initializer_base::GetParameter(const Vector3D<S32>& i_inputStruct, U32 i_inputCH, const Vector3D<S32>& i_outputStruct, U32 i_outputCH, U32 i_parameterCount, F32 o_lpParameter[])
{
	return this->GetParameter(
		IODataStruct(i_inputCH,  i_inputStruct.x,  i_inputStruct.y,  i_inputStruct.z),
		IODataStruct(i_outputCH, i_outputStruct.x, i_outputStruct.y, i_outputStruct.z),
		i_parameterCount, o_lpParameter);
}