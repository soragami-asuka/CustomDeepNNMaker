//======================================
// �o�͐M���������C���[
// GPU�����p
//======================================
#ifndef __SignalArray2Value_GPU_H__
#define __SignalArray2Value_GPU_H__

#pragma warning(push)
#pragma warning(disable : 4267)
#pragma warning(disable : 4819)
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)

#include"SignalArray2Value_DATA.hpp"
#include"SignalArray2Value_FUNC.hpp"
#include"SignalArray2Value_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

class SignalArray2Value_GPU : public SignalArray2Value_Base
{
private:
	// �f�[�^�{��
	class SignalArray2Value_LayerData_GPU& layerData;

	// ���o�̓o�b�t�@


	// Get�֐����g���Ə������ׂ������ނ̂ňꎞ�ۑ��p. PreCalculate�Œl���i�[.
	U32 inputBufferCount;				/**< ���̓o�b�t�@�� */
	U32 outputBufferCount;				/**< �o�̓o�b�t�@�� */

	U32 inputBatchBufferSize;			/**< ���͐M���̃o�b�`���Ƃ̃o�b�t�@�T�C�Y */
	U32 inputChannelSize;				/**< ���͐M���̃`�����l�����Ƃ̃o�b�t�@�T�C�Y */

	std::vector<F32> lpTmpOutputBuffer_h;	/**< �ꎞ�o�̓o�b�t�@(�z�X�g������) */
	std::vector<F32*> lpTmpBatchOutputBuffer_h;	/**< �o�b�`���Ƃ̈ꎞ�o�̓o�b�t�@(�z�X�g������) */

	cublasHandle_t cublasHandle;

public:
	/** �R���X�g���N�^ */
	SignalArray2Value_GPU(Gravisbell::GUID guid, class SignalArray2Value_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
	/** �f�X�g���N�^ */
	virtual ~SignalArray2Value_GPU();

public:
	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 GetLayerKind()const;

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Initialize(void);


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	SignalArray2Value_LayerData_Base& GetLayerData();
	const SignalArray2Value_LayerData_Base& GetLayerData()const;


public:
	//================================
	// ���Z����
	//================================
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessLearn();

	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessCalculate();


	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessLoop();


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer);

public:
	//================================
	// �w�K����
	//================================
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v�Ȕz���[GetOutputDataCount()]�z��
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);
};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif