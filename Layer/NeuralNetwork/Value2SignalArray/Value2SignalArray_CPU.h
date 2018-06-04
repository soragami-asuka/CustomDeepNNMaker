//======================================
// �o�͐M���������C���[
// CPU�����p
//======================================
#ifndef __Value2SignalArray_CPU_H__
#define __Value2SignalArray_CPU_H__

#include"stdafx.h"

#include"Value2SignalArray_DATA.hpp"
#include"Value2SignalArray_FUNC.hpp"
#include"Value2SignalArray_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

class Value2SignalArray_CPU : public Value2SignalArray_Base
{
private:
	// �f�[�^�{��
	class Value2SignalArray_LayerData_CPU& layerData;

	// ���o�̓o�b�t�@

	// Get�֐����g���Ə������ׂ������ނ̂ňꎞ�ۑ��p. PreCalculate�Œl���i�[.
	U32 inputBufferCount;				/**< ���̓o�b�t�@�� */
	U32 outputBufferCount;				/**< �o�̓o�b�t�@�� */
	U32 channelSize;					/**< �`�����l���̎��o�b�t�@�T�C�Y */

	// ���o�̓o�b�t�@
	std::vector<CONST_BATCH_BUFFER_POINTER> lppBatchInputBuffer;	/**< ���Z���̓��̓f�[�^ */
	std::vector<BATCH_BUFFER_POINTER>		lppBatchOutputBuffer;	/**< �o�b�`�����p�o�̓o�b�t�@ <�o�b�`��> */
	std::vector<BATCH_BUFFER_POINTER>		lppBatchDInputBuffer;	/**< ���͌덷�v�Z���̏o�͌덷�f�[�^ */
	std::vector<CONST_BATCH_BUFFER_POINTER> lppBatchDOutputBuffer;	/**< ���͌덷�v�Z���̏o�͌덷�f�[�^ */

public:
	/** �R���X�g���N�^ */
	Value2SignalArray_CPU(Gravisbell::GUID guid, class Value2SignalArray_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
	/** �f�X�g���N�^ */
	virtual ~Value2SignalArray_CPU();

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
	Value2SignalArray_LayerData_Base& GetLayerData();
	const Value2SignalArray_LayerData_Base& GetLayerData()const;


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