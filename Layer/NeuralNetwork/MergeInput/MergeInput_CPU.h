//======================================
// �����֐����C���[
// CPU�����p
//======================================
#ifndef __MergeInput_CPU_H__
#define __MergeInput_CPU_H__

#include"stdafx.h"

#include"MergeInput_DATA.hpp"
#include"MergeInput_FUNC.hpp"
#include"MergeInput_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

class MergeInput_CPU : public MergeInput_Base
{
private:
	// �f�[�^�{��
	class MergeInput_LayerData_CPU& layerData;

	// ���o�̓o�b�t�@
	std::vector<F32>				lpOutputBuffer;		/**< �o�̓o�b�t�@ [�o�b�`�� * �o�͐M����] */
	std::vector<std::vector<F32>>	lpDInputBuffer;		/**< ���͌덷���� [���̓f�[�^��][�o�b�`�� * ���͐M����] */

	std::vector<F32*>				lppBatchOutputBuffer;		/**< �o�b�`�����p�o�̓o�b�t�@ <�o�b�`��> */
	std::vector<std::vector<F32*>>	lppBatchDInputBuffer;		/**< �o�b�`�����p���͌덷�o�b�t�@ <���̓f�[�^��><�o�b�`��> */


	// Get�֐����g���Ə������ׂ������ނ̂ňꎞ�ۑ��p. PreCalculate�Œl���i�[.
	std::vector<U32>	lpInputBufferCount;		/**< ���̓o�b�t�@�� */
	U32					outputBufferCount;		/**< �o�̓o�b�t�@�� */

	// ���Z���̓��̓f�[�^
	std::vector<std::vector<CONST_BATCH_BUFFER_POINTER>>	m_lppInputBuffer;		/**< ���Z���̓��̓f�[�^[���̓f�[�^��][�o�b�`�T�C�Y] */
	std::vector<CONST_BATCH_BUFFER_POINTER>					m_lppDOutputBufferPrev;	/**< ���͌덷�v�Z���̏o�͌덷�f�[�^ */


public:
	/** �R���X�g���N�^ */
	MergeInput_CPU(Gravisbell::GUID guid, class MergeInput_LayerData_CPU& i_layerData);
	/** �f�X�g���N�^ */
	virtual ~MergeInput_CPU();

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
	MergeInput_LayerData_Base& GetLayerData();
	const MergeInput_LayerData_Base& GetLayerData()const;


public:
	//================================
	// ���Z����
	//================================
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessLearn(unsigned int batchSize);

	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessCalculate(unsigned int batchSize);


	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& data);
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessCalculateLoop();


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[]);

	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const;
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const;

public:
	//================================
	// �w�K����
	//================================
	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER GetDInputBuffer(U32 i_dataNum)const;
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode GetDInputBuffer(U32 i_dataNum, BATCH_BUFFER_POINTER o_lpDInputBuffer)const;

};


} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif